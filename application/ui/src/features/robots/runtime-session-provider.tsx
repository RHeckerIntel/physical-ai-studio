import { createContext, ReactNode, RefObject, useContext, useRef, useState } from 'react';

import { useMutation, UseMutationResult, useQueryClient } from '@tanstack/react-query';

import {
    SchemaDatasetOutput,
    SchemaEnvironmentWithRelations,
    SchemaInferenceDeviceInfo,
    SchemaModel,
} from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { useProjectId } from '../projects/use-project';
import { FollowerSource } from './use-joint-state';
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

/** One feature's current value, as the runtime streams it. */
interface Sample {
    value: number;
    timestamp: number;
}

/**
 * Split the runtime's features into the observation and action maps the pages
 * above expect, keyed by joint.
 *
 * The runtime keys features by kind and robot -- ``observation.arm.wrist.pos``
 * -- while these pages index by joint alone, as a recorded dataset does. Only
 * ``driven`` robots are included: a leader's joints carry the same names, so
 * including them would overwrite the arm being watched with the one being held.
 */
const splitFeatures = (data: Record<string, Sample>, driven: Set<string>) => {
    const observation: Record<string, number> = {};
    const actions: Record<string, number> = {};
    for (const [key, sample] of Object.entries(data)) {
        const parts = key.split('.');
        if (parts.length < 4 || !driven.has(parts[1])) {
            continue;
        }
        const target = parts[0] === 'action' ? actions : parts[0] === 'observation' ? observation : undefined;
        if (target !== undefined) {
            target[parts.slice(2).join('.')] = sample.value;
        }
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

interface RuntimeSessionProviderProps {
    children: ReactNode;
    environment: SchemaEnvironmentWithRelations;
    model?: SchemaModel;
    dataset?: SchemaDatasetOutput;
    inferenceDevice?: InferenceDevice;
    onError: (error: string) => void;
}

type MutationResult<TVariables = void> = UseMutationResult<RuntimeApiJsonResponse, Error, TVariables>;

type RuntimeSessionContextValue = {
    observation: RefObject<Record<string, number> | undefined>;
    actions: RefObject<Record<string, number> | undefined>;
    environment: SchemaEnvironmentWithRelations;
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

    const onOpen = () => {
        // The session opens empty, so loading is a command rather than part of
        // the URL. Everything else waits until the devices are up.
        void (async () => {
            try {
                await loadEnvironment.mutateAsync(props.environment.id);
            } catch {
                return; // already reported through onError
            }
            if (props.model && props.inferenceDevice) {
                loadModel.mutate({ model: props.model, inference_device: props.inferenceDevice });
            }
            if (props.dataset) {
                loadDataset.mutate(props.dataset);
                setFollowerSource.mutate('teleop');
            }
        })();
    };

    const socket = useWebSocketWithResponse(runtimeV2SocketUrl(project_id), {
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
            }
            if (message.event === 'error') {
                props.onError(typeof message.message === 'string' ? message.message : 'An unexpected error occurred.');
            }
        },
        onError: console.error,
        onClose: () => {
            setState(createRuntimeSessionState());
        },
        onOpen,
    });

    const loadEnvironment = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (environment_id: string) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'load_environment', environment_id },
                ({ event, data }) => event === 'state' && data?.loaded === true
            ),
        onError: (error: Error) => props.onError(error.message),
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
