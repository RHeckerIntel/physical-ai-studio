import { useCallback, useMemo, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';

export interface RuntimeV2Feature {
    key: string;
    value: number;
    timestamp: number;
}

export interface RuntimeV2State {
    /** False before the first load and after an unload, which are normal states. */
    loaded: boolean;
    environment: string | null;
    robots: Record<string, string>;
    teleoperating: boolean;
    features: number;
}

export type ParsedRuntimeV2Message =
    | { type: 'state'; state: RuntimeV2State }
    | { type: 'observation'; features: RuntimeV2Feature[] }
    | { type: 'error'; message: string; errorCode: string }
    | { type: 'ignored' };

/**
 * The v2 socket sends whole-store snapshots rather than a fixed observation
 * shape, so every key arrives with its own timestamp and unauthored features
 * are simply absent. Parsing is kept separate from the hook so it can be
 * tested without a socket.
 */
export const parseRuntimeV2Message = (payload: unknown): ParsedRuntimeV2Message => {
    if (typeof payload !== 'object' || payload === null || !('event' in payload)) {
        return { type: 'ignored' };
    }

    const message = payload as { event?: unknown; data?: unknown; message?: unknown; error_code?: unknown };

    if (message.event === 'state' && typeof message.data === 'object' && message.data !== null) {
        return { type: 'state', state: message.data as RuntimeV2State };
    }

    if (message.event === 'observation' && typeof message.data === 'object' && message.data !== null) {
        const entries = Object.entries(message.data as Record<string, { value: number; timestamp: number }>);
        return {
            type: 'observation',
            features: entries.map(([key, sample]) => ({
                key,
                value: Number(sample.value),
                timestamp: Number(sample.timestamp),
            })),
        };
    }

    if (message.event === 'error') {
        return {
            type: 'error',
            message: typeof message.message === 'string' ? message.message : 'Failed to load the environment.',
            errorCode: typeof message.error_code === 'string' ? message.error_code : 'runtime_session_failed',
        };
    }

    return { type: 'ignored' };
};

// Composed from a typed project path; the v2 socket is not in OpenAPI yet.
export const runtimeV2SocketUrl = (project_id: string): string =>
    `${fetchClient.PATH('/api/projects/{project_id}', { params: { path: { project_id } } })}/runtimev2/ws`;

/**
 * Hold a runtime v2 session open for as long as this hook is mounted.
 *
 * The session and the environment have separate lifetimes. Opening the socket
 * creates an empty session; `environment_id` is loaded into it immediately, and
 * can be unloaded or replaced without reconnecting. Unmounting closes the
 * socket, which unloads whatever is still loaded and releases its robots.
 */
export const useRuntimeV2Session = (project_id: string, environment_id: string) => {
    const [state, setState] = useState<RuntimeV2State | null>(null);
    const [features, setFeatures] = useState<RuntimeV2Feature[]>([]);
    const [error, setError] = useState<string | null>(null);
    const hasFailed = useRef(false);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        const parsed = parseRuntimeV2Message(JSON.parse(event.data));

        if (parsed.type === 'state') {
            setState(parsed.state);
            setError(null);
        } else if (parsed.type === 'observation') {
            setFeatures(parsed.features);
        } else if (parsed.type === 'error') {
            // A failed command leaves the session open, so this is not fatal to
            // the socket -- only to whatever was asked for.
            hasFailed.current = true;
            setError(parsed.message);
        }
    }, []);

    const { sendJsonMessage, readyState } = useWebSocket(runtimeV2SocketUrl(project_id), {
        onOpen: () => {
            hasFailed.current = false;
            setError(null);
            // The session opens empty; loading is a command like any other, so
            // the same socket can swap environments without reconnecting.
            sendJsonMessage({ event: 'load_environment', environment_id });
        },
        onClose: () => {
            setState(null);
            setFeatures([]);
        },
        onMessage: handleMessage,
        // A failed load is a real answer, not a blip: retrying would reconnect
        // the robots behind the user's back.
        shouldReconnect: () => !hasFailed.current,
        retryOnError: false,
    });

    const setTeleoperating = useCallback(
        (enabled: boolean) => {
            sendJsonMessage({ event: 'set_teleoperating', enabled });
        },
        [sendJsonMessage]
    );

    const loadEnvironment = useCallback(
        (id: string) => {
            sendJsonMessage({ event: 'load_environment', environment_id: id });
        },
        [sendJsonMessage]
    );

    const unloadEnvironment = useCallback(() => {
        sendJsonMessage({ event: 'unload_environment' });
    }, [sendJsonMessage]);

    const grouped = useMemo(() => groupFeatures(features), [features]);

    return {
        state,
        features,
        grouped,
        error,
        isConnecting: (state === null || !state.loaded) && error === null,
        isOpen: readyState === 1,
        setTeleoperating,
        loadEnvironment,
        unloadEnvironment,
    };
};

export interface FeatureGroup {
    robot: string;
    joints: Array<{ joint: string; observation?: number; action?: number }>;
}

/**
 * Pair each joint's observation with the action commanded for it.
 *
 * Seeing them side by side is the point of the screen: with teleoperation off
 * the action column shows what *would* be sent, so the follower can be watched
 * before it is allowed to move.
 */
export const groupFeatures = (features: RuntimeV2Feature[]): FeatureGroup[] => {
    const robots = new Map<string, Map<string, { observation?: number; action?: number }>>();

    for (const feature of features) {
        // observation.<robot>.<joint>.pos / action.<robot>.<joint>.pos
        const parts = feature.key.split('.');
        if (parts.length !== 4 || parts[3] !== 'pos') continue;
        const [kind, robot, joint] = parts;
        if (kind !== 'observation' && kind !== 'action') continue;

        const joints = robots.get(robot) ?? new Map();
        robots.set(robot, joints);
        const entry = joints.get(joint) ?? {};
        joints.set(joint, { ...entry, [kind]: feature.value });
    }

    return [...robots.entries()].map(([robot, joints]) => ({
        robot,
        joints: [...joints.entries()].map(([joint, values]) => ({ joint, ...values })),
    }));
};
