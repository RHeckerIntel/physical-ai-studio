import {
    createContext,
    ReactNode,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    useSyncExternalStore,
} from 'react';

import useWebSocket from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

import { fetchClient } from '../../api/client';
import {
    SchemaAckMessage,
    SchemaEnvironmentDraft,
    SchemaErrorMessage,
    SchemaLoadEnvironmentCommand,
    SchemaObservationMessage,
    SchemaSessionState,
    SchemaSetControlCommand,
    SchemaStateMessage,
    SchemaUnloadEnvironmentCommand,
} from '../../api/openapi-spec';
import { FrameSource } from '../cameras/camera-view';
import { JointsState } from '../robots/use-robot-observations';

type JsonServerMessage = SchemaAckMessage | SchemaStateMessage | SchemaErrorMessage;
type Command =
    | Omit<SchemaLoadEnvironmentCommand, 'request_id'>
    | Omit<SchemaUnloadEnvironmentCommand, 'request_id'>
    | Omit<SchemaSetControlCommand, 'request_id'>;

type Listener<T> = (value: T) => void;

/** Latest value plus listeners, so 20 Hz data reaches its subscribers without React state. */
class Channel<T> {
    value: T | undefined = undefined;
    private listeners = new Set<Listener<T>>();

    publish(value: T) {
        this.value = value;
        this.listeners.forEach((listener) => listener(value));
    }

    subscribe(listener: Listener<T>) {
        this.listeners.add(listener);
        return () => {
            this.listeners.delete(listener);
        };
    }
}

interface RuntimeSessionContextValue {
    state: SchemaSessionState | null;
    /** The last command or worker error, until the next successful load. */
    error: SchemaErrorMessage | null;
    isLoading: boolean;
    observations: Channel<SchemaObservationMessage>;
    frame: (key: string) => Channel<Blob>;
    send: (command: Command) => Promise<SchemaAckMessage>;
}

const RuntimeSessionContext = createContext<RuntimeSessionContextValue | null>(null);

const SESSION_HZ = 20;

const sessionSocketUrl = (project_id: string) =>
    fetchClient.PATH('/api/projects/{project_id}/runtime/ws', { params: { path: { project_id } } });

/** ``[u32 big-endian header length][header JSON][jpeg bytes, in the order of header.frames]`` */
export const parseObservation = async (
    blob: Blob
): Promise<{ observation: SchemaObservationMessage; frames: Map<string, Blob> }> => {
    const length = new DataView(await blob.slice(0, 4).arrayBuffer()).getUint32(0);
    const observation = JSON.parse(await blob.slice(4, 4 + length).text()) as SchemaObservationMessage;
    const frames = new Map<string, Blob>();
    let offset = 4 + length;
    for (const { key, size } of observation.frames) {
        frames.set(key, blob.slice(offset, offset + size, 'image/jpeg'));
        offset += size;
    }
    return { observation, frames };
};

const commandError = (ack: SchemaAckMessage): SchemaErrorMessage => ({
    type: 'error',
    component: 'session',
    message: ack.error?.message ?? 'The command failed.',
    error_code: ack.error?.error_code ?? 'unknown',
    session_alive: true,
});

/**
 * One runtime session for as long as this provider is mounted.
 *
 * Loads ``environment`` into the session and reloads it whenever it changes. Pages
 * read joints and frames from here and never open their own device sockets.
 */
export const RuntimeSessionProvider = ({
    project_id,
    environment,
    children,
}: {
    project_id: string;
    environment: SchemaEnvironmentDraft | null;
    children: ReactNode;
}) => {
    const [state, setState] = useState<SchemaSessionState | null>(null);
    const [error, setError] = useState<SchemaErrorMessage | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const observations = useRef(new Channel<SchemaObservationMessage>()).current;
    const frames = useRef(new Map<string, Channel<Blob>>()).current;
    const pending = useRef(new Map<string, (ack: SchemaAckMessage) => void>()).current;
    const sessionClosed = useRef(false);

    const frame = useCallback(
        (key: string) => {
            let channel = frames.get(key);
            if (channel === undefined) {
                channel = new Channel<Blob>();
                frames.set(key, channel);
            }
            return channel;
        },
        [frames]
    );

    const { sendJsonMessage } = useWebSocket(sessionSocketUrl(project_id), {
        queryParams: { hz: SESSION_HZ },
        // A new socket is a new session; the load effect below loads it again.
        shouldReconnect: () => !sessionClosed.current,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onClose: () => {
            setState(null);
            pending.forEach((resolve, request_id) =>
                resolve({
                    type: 'ack',
                    request_id,
                    ok: false,
                    error: { message: 'Connection closed', error_code: 'connection_closed' },
                })
            );
            pending.clear();
        },
        onMessage: (event: MessageEvent) => {
            if (event.data instanceof Blob) {
                void parseObservation(event.data).then(({ observation, frames: jpegs }) => {
                    observations.publish(observation);
                    jpegs.forEach((jpeg, key) => frame(key).publish(jpeg));
                });
                return;
            }
            const message = JSON.parse(event.data) as JsonServerMessage;
            switch (message.type) {
                case 'state':
                    setState(message.data);
                    break;
                case 'error':
                    sessionClosed.current = !message.session_alive;
                    setError(message);
                    break;
                case 'ack':
                    pending.get(message.request_id)?.(message);
                    pending.delete(message.request_id);
                    break;
            }
        },
    });

    const send = useCallback(
        (command: Command) =>
            new Promise<SchemaAckMessage>((resolve) => {
                const request_id = uuidv4();
                pending.set(request_id, resolve);
                sendJsonMessage({ ...command, request_id });
            }),
        [pending, sendJsonMessage]
    );

    // Reload whenever the environment changes, and on every new socket (a new session).
    const environmentKey = JSON.stringify(environment);
    const hasSession = state !== null;
    useEffect(() => {
        if (!hasSession) {
            return;
        }
        let cancelled = false;
        const draft = JSON.parse(environmentKey) as SchemaEnvironmentDraft | null;
        const reload = async () => {
            setIsLoading(true);
            await send({ command: 'unload_environment' });
            if (draft !== null && draft.robots.length > 0 && !cancelled) {
                const ack = await send({ command: 'load_environment', environment: draft });
                if (!cancelled) {
                    setError(ack.ok ? null : commandError(ack));
                }
            }
            if (!cancelled) {
                setIsLoading(false);
            }
        };
        void reload();
        return () => {
            cancelled = true;
        };
    }, [environmentKey, hasSession, send]);

    const value = useMemo(
        () => ({ state, error, isLoading, observations, frame, send }),
        [state, error, isLoading, observations, frame, send]
    );

    return <RuntimeSessionContext.Provider value={value}>{children}</RuntimeSessionContext.Provider>;
};

export const useRuntimeSession = () => {
    const context = useContext(RuntimeSessionContext);
    if (context === null) {
        throw new Error('useRuntimeSession must be used within a RuntimeSessionProvider');
    }
    return context;
};

export const useOptionalRuntimeSession = () => useContext(RuntimeSessionContext);

/** The session robot's measured joints, re-rendering the caller at the session rate. */
export const useSessionJoints = (): JointsState => {
    const { observations } = useRuntimeSession();
    const observation = useSyncExternalStore(
        (onChange) => observations.subscribe(onChange),
        () => observations.value
    );
    return useMemo(
        () => Object.entries(observation?.state ?? {}).map(([name, value]) => ({ name, value })),
        [observation]
    );
};

/** Frames of one session camera, by store key. */
export const useSessionCamera = (key: string): FrameSource => {
    const { frame } = useRuntimeSession();
    return frame(key);
};
