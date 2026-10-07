import { createContext, ReactNode, RefObject, useContext, useEffect, useRef, useState } from 'react';

import { useMutation, UseMutationResult, useQueryClient } from '@tanstack/react-query';

import {
    SchemaDatasetOutput,
    SchemaEnvironmentWithRelations,
    SchemaInferenceDeviceInfo,
    SchemaModel,
} from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { useProjectId } from '../projects/use-project';
import { FollowerSource, isRecoverableRobotControlError } from './use-joint-state';
import { runtimeV2SocketUrl } from './use-runtimev2-session';

type InferenceDevice = Pick<SchemaInferenceDeviceInfo, 'backend' | 'device'>;

export interface RuntimeSessionState {
    connected: boolean;
    follower_source: FollowerSource;
    has_leader: boolean;
    model_loaded: boolean;
    task: string | null;
    dataset_loaded: boolean;
    is_recording: boolean;
    episodes_recorded: number;
}

const createRuntimeSessionState = (): RuntimeSessionState => ({
    connected: false,
    follower_source: 'hold',
    has_leader: false,
    model_loaded: false,
    task: null,
    dataset_loaded: false,
    is_recording: false,
    episodes_recorded: 0,
});

interface RuntimeApiJsonResponse<T = RuntimeV2State> {
    event: string;
    data?: T;
    message?: string;
    error_code?: string;
}

/** What a control is, as the runtime reports and accepts it. */
interface ControlConfig {
    kind: 'teleop' | 'model';
    hz?: number;
    task?: string | null;
}

/**
 * The runtime's own state. Mapped onto {@link RuntimeSessionState} so the
 * pages above this provider keep the vocabulary they were written against.
 */
interface RuntimeV2State {
    loaded: boolean;
    environment: string | null;
    robots: Record<string, string>;
    leaders: string[];
    cameras: string[];
    features: number;
    control: ControlConfig | null;
    task: string | null;
    dataset_loaded: boolean;
    dataset_id: string | null;
    is_recording: boolean;
    episodes_recorded: number;
    model_loaded: boolean;
    model_id: string | null;
    model_chunk_size: number | null;
}

/**
 * Which control corresponds to each follower source.
 *
 * `hold` is the absence of a control: nothing writes the action features, so
 * an arm keeps the position it was last commanded to.
 */
const controlFor = (source: FollowerSource): ControlConfig | null => {
    if (source === 'teleop') return { kind: 'teleop' };
    if (source === 'policy') return { kind: 'model' };
    return null;
};

const sourceOf = (control: ControlConfig | null): FollowerSource => {
    if (control?.kind === 'teleop') return 'teleop';
    if (control?.kind === 'model') return 'policy';
    return 'hold';
};

/** One feature's current vector, as the runtime streams it. */
interface Sample {
    names: string[];
    values: number[];
    timestamp: number;
}

/**
 * Split the runtime's features into the observation and action maps the pages
 * above expect, keyed by joint.
 *
 * A feature is a robot's whole vector with its component names, so this zips
 * rather than parsing keys -- and the order is the driver's, which is the
 * order that matters. Only `driven` robots are included: a leader's joints
 * carry the same names, so including them would overwrite the arm being
 * watched with the one being held.
 */
const splitFeatures = (data: Record<string, Sample>, driven: Set<string>) => {
    const observation: Record<string, number> = {};
    const actions: Record<string, number> = {};
    for (const [key, sample] of Object.entries(data)) {
        const [kind, robot] = key.split('.');
        if (!driven.has(robot)) {
            continue;
        }
        const target = kind === 'action' ? actions : kind === 'observation' ? observation : undefined;
        if (target === undefined) {
            continue;
        }
        sample.names.forEach((name, index) => {
            target[`${name}.pos`] = sample.values[index];
        });
    }
    return { observation, actions };
};

const toSessionState = (next: Partial<RuntimeV2State>): RuntimeSessionState => ({
    connected: next.loaded ?? false,
    follower_source: sourceOf(next.control ?? null),
    has_leader: (next.leaders?.length ?? 0) > 0,
    model_loaded: next.model_loaded ?? false,
    task: next.task ?? null,
    dataset_loaded: next.dataset_loaded ?? false,
    is_recording: next.is_recording ?? false,
    episodes_recorded: next.episodes_recorded ?? 0,
});

/** An explicit set of devices, for a session with no saved environment. */
export interface SessionDevices {
    follower_id: string;
    leader_id?: string;
    camera_ids: string[];
}

interface RuntimeSessionProviderProps {
    children: ReactNode;
    /** A saved environment to open, or `devices` for one that is not saved. */
    environment?: SchemaEnvironmentWithRelations;
    /** Devices to open directly: the environment form previews robots the
     * user is still choosing, so there is nothing saved to load. */
    devices?: SessionDevices;
    model?: SchemaModel;
    dataset?: SchemaDatasetOutput;
    inferenceDevice?: InferenceDevice;
    onError?: (error: string) => void;
}

type MutationResult<TVariables = void> = UseMutationResult<RuntimeApiJsonResponse, Error, TVariables>;

type RuntimeSessionContextValue = {
    observation: RefObject<Record<string, number> | undefined>;
    actions: RefObject<Record<string, number> | undefined>;
    /** Undefined when the session was opened from `devices`. */
    environment: SchemaEnvironmentWithRelations | undefined;
    model: SchemaModel | undefined;
    dataset: SchemaDatasetOutput | undefined;
    inferenceDevice: InferenceDevice | undefined;
    state: RuntimeSessionState;
    loadModel: MutationResult<{ model: SchemaModel; inference_device: InferenceDevice }>;
    loadDataset: MutationResult<SchemaDatasetOutput>;
    startTask: MutationResult<string>;
    stopTask: MutationResult;
    setFollowerSource: MutationResult<FollowerSource>;
    startEpisode: MutationResult<string>;
    saveEpisode: MutationResult;
    discardEpisode: MutationResult;
    readyForInference: boolean;
    readyForRecording: boolean;
    isConnected: boolean;
    /** The last fatal error and its code. A recoverable one is a `warning`
     * instead, so a view keeps rendering rather than being replaced. */
    error: string | null;
    errorCode: string | null;
    warning: string | null;
    /** Reconnect, which ends the old session and opens a new one. */
    restart: () => void;
};

const RuntimeSessionContext = createContext<RuntimeSessionContextValue | null>(null);

const EPISODE_ACK_TIMEOUT_MS = 60_000;

const useRefreshEpisodes = (dataset_id?: string) => {
    const queryClient = useQueryClient();

    return () => {
        if (dataset_id === undefined) {
            return;
        }
        queryClient.invalidateQueries({
            queryKey: [
                'get',
                '/api/dataset/{dataset_id}/episodes',
                {
                    params: { path: { dataset_id } },
                },
            ],
        });
    };
};

export const RuntimeSessionProvider = (props: RuntimeSessionProviderProps) => {
    const { project_id } = useProjectId();
    const [state, setState] = useState<RuntimeSessionState>(createRuntimeSessionState());
    const observation = useRef<Record<string, number> | undefined>(undefined);
    const actions = useRef<Record<string, number> | undefined>(undefined);
    // Which robots this session drives, so a leader's identically named joints
    // are not mistaken for theirs. Filled from each state message.
    const driven = useRef<Set<string>>(new Set());
    const [model, setModel] = useState<SchemaModel | undefined>(props.model);
    const [inferenceDevice, setInferenceDevice] = useState<InferenceDevice | undefined>(props.inferenceDevice);
    const [dataset, setDataset] = useState<SchemaDatasetOutput | undefined>(props.dataset);
    const invalidateEpisodesQuery = useRefreshEpisodes(dataset?.id);
    const [error, setError] = useState<string | null>(null);
    const [errorCode, setErrorCode] = useState<string | null>(null);
    const [warning, setWarning] = useState<string | null>(null);
    // Changing the URL tears the socket down and opens a new one, which is
    // what ends a session and starts a fresh one. A query parameter rather
    // than a fragment: a WebSocket URL may not have one, and the constructor
    // throws if it does.
    const [attempt, setAttempt] = useState(0);

    const report = (message: string, code: string | null = null) => {
        // A lost leader is worth saying and worth recovering from, so it does
        // not replace the view the way a dead session does.
        if (isRecoverableRobotControlError(code)) {
            setWarning(message);
            return;
        }
        setError(message);
        setErrorCode(code);
        props.onError?.(message);
    };

    const onOpen = () => {
        // The session opens empty, so loading is a command rather than part of
        // the URL. Everything else waits until the devices are up.
        void (async () => {
            setError(null);
            setErrorCode(null);
            setWarning(null);
            try {
                await openSession.mutateAsync();
            } catch {
                return; // already reported
            }
            if (props.model && props.inferenceDevice) {
                loadModel.mutate({ model: props.model, inference_device: props.inferenceDevice });
            }
            if (props.dataset) {
                loadDataset.mutate(props.dataset);
                //setFollowerSource.mutate('teleop');
            }
        })();
    };

    const socket = useWebSocketWithResponse(`${runtimeV2SocketUrl(project_id)}?attempt=${attempt}`, {
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: (event: WebSocketEventMap['message']) => {
            const message = JSON.parse(event.data) as RuntimeApiJsonResponse<unknown>;
            if (message.event === 'observation' && message.data !== undefined && typeof message.data === 'object') {
                const split = splitFeatures(message.data as Record<string, Sample>, driven.current);
                observation.current = split.observation;
                // Undefined rather than empty until something is driving, so a
                // viewer falls back to what the arm measured.
                actions.current = Object.keys(split.actions).length > 0 ? split.actions : undefined;
            }
            if (message.event === 'state' && message.data !== undefined && typeof message.data === 'object') {
                const next = message.data as Partial<RuntimeV2State>;
                driven.current = new Set(Object.keys(next.robots ?? {}));
                setState(toSessionState(next));
                setWarning(null);
            }
            if (message.event === 'error') {
                report(
                    typeof message.message === 'string' ? message.message : 'An unexpected error occurred.',
                    typeof message.error_code === 'string' ? message.error_code : null
                );
            }
        },
        onError: console.error,
        onClose: () => {
            setState(createRuntimeSessionState());
        },
        onOpen,
    });

    /** Open whichever source this provider was given.
     *
     * An environment is loaded by id. Devices are named explicitly, which is
     * what the environment form needs: it previews robots the user is still
     * choosing, so there is nothing saved to load.
     */
    const openSession = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () => {
            const message =
                props.environment !== undefined
                    ? { event: 'load_environment', environment_id: props.environment.id }
                    : { event: 'load_devices', ...props.devices };
            return socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                message,
                ({ event, data }) => event === 'state' && data?.loaded === true
            );
        },
        onError: (failure: Error) => report(failure.message, 'runtime_session_failed'),
    });

    const loadModel = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (properties: { model: SchemaModel; inference_device: InferenceDevice }) => {
            const result = await socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                {
                    event: 'load_model',
                    model_id: properties.model.id,
                    inference_device: properties.inference_device,
                },
                ({ event, data }) => event === 'state' && data?.model_loaded === true
            );
            setModel(properties.model);
            setInferenceDevice(properties.inference_device);
            return result;
        },
    });

    const loadDataset = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (datasetConfig: SchemaDatasetOutput) => {
            const result = await socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'load_dataset', dataset_id: datasetConfig.id },
                ({ event, data }) => event === 'state' && data?.dataset_loaded === true
            );
            setDataset(datasetConfig);
            return result;
        },
    });

    // Starting a task is choosing the policy and the instruction together: the
    // task travels with the control, so a new one resets the policy rather than
    // leaving it acting on a chunk predicted for the old instruction.
    const startTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'set_control', control: { kind: 'model', task } },
                ({ event, data }) => event === 'state' && data?.control?.kind === 'model'
            ),
    });

    // Stopping it halts the policy. Nothing writes the action features after
    // that, so the arm holds the position it was last commanded to.
    const stopTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'set_control', control: null },
                ({ event, data }) => event === 'state' && data?.control === null
            ),
    });

    const setFollowerSource = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (follower_source: FollowerSource) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'set_control', control: controlFor(follower_source) },
                ({ event, data }) => event === 'state' && sourceOf(data?.control ?? null) === follower_source
            ),
    });

    const startEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'start_recording', task },
                ({ event, data }) => event === 'state' && data?.is_recording === true
            ),
    });

    const saveEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () => {
            const result = await socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'save_episode' },
                undefined,
                { timeout: EPISODE_ACK_TIMEOUT_MS }
            );
            invalidateEpisodesQuery();
            return result;
        },
    });

    const discardEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>({ event: 'discard_episode' }, undefined, {
                timeout: EPISODE_ACK_TIMEOUT_MS,
            }),
    });

    return (
        <RuntimeSessionContext.Provider
            value={{
                observation,
                actions,
                environment: props.environment,
                model,
                dataset,
                inferenceDevice,
                state,
                loadModel,
                loadDataset,
                startTask,
                stopTask,
                setFollowerSource,
                startEpisode,
                saveEpisode,
                discardEpisode,
                readyForInference: state.connected && state.model_loaded,
                readyForRecording: state.connected && state.dataset_loaded,
                isConnected: socket.readyState === 1,
                error,
                errorCode,
                warning,
                restart: () => setAttempt((previous) => previous + 1),
            }}
        >
            {props.children}
        </RuntimeSessionContext.Provider>
    );
};

export const useRuntimeSession = () => {
    const ctx = useContext(RuntimeSessionContext);
    if (!ctx) throw new Error('useRuntimeSession must be used within RuntimeSessionProvider');
    return ctx;
};

/**
 * Sample the session's observations into state, for a view that renders them.
 *
 * The session keeps observations in a ref so a 100Hz robot does not re-render
 * the page a hundred times a second. A component that draws a model needs
 * React to see the change, so it samples here instead -- on animation frames,
 * which stop when the tab is hidden and never outpace the display.
 */
export const useSessionJoints = (): Array<{ name: string; value: number }> => {
    const { observation } = useRuntimeSession();
    const [joints, setJoints] = useState<Array<{ name: string; value: number }>>([]);

    useEffect(() => {
        let frame = 0;
        const sample = () => {
            const current = observation.current;
            if (current !== undefined) {
                setJoints(Object.entries(current).map(([name, value]) => ({ name, value: Number(value) })));
            }
            frame = requestAnimationFrame(sample);
        };
        frame = requestAnimationFrame(sample);
        return () => cancelAnimationFrame(frame);
    }, [observation]);

    return joints;
};
