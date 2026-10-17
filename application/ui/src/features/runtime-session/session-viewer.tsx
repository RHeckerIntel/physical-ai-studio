import { ReactNode, Suspense, useEffect, useRef, useState } from 'react';

import { Button, Flex, Heading, Loading, ProgressCircle, StatusLight, Text, View } from '@geti-ui/ui';
import {
    DockviewApi,
    DockviewReact,
    DockviewReadyEvent,
    IDockviewPanelProps,
    IDockviewReactProps,
} from 'dockview-react';

import { SchemaSessionEnvironment } from '../../api/openapi-spec';
import { physicalAiTheme } from '../dockview';
import { InlineAlert } from '../robots/setup-wizard/shared/inline-alert';
import { CameraCell } from './cells/camera-cell';
import { RobotCell } from './cells/robot-cell';
import { failureOf, useRuntimeSession } from './runtime-session-provider';

/** Something a page waits for before its session is usable, such as the environment or a dataset. */
export interface SessionPart {
    label: string;
    isLoaded: boolean;
    /** Why it failed; it stays unloaded until it is loaded again. */
    error?: string | null;
}

const components = {
    // The workspace's side bar has the Teleoperate switch.
    follower: (props: IDockviewPanelProps<{ robot_id: string }>) => (
        <RobotCell follower_id={props.params.robot_id} showTeleop={false} />
    ),
    camera: (props: IDockviewPanelProps<{ camera_id: string }>) => <CameraCell camera_id={props.params.camera_id} />,
} satisfies IDockviewReactProps['components'];

/** One panel per camera and one for the follower, keeping panels that are still part of the environment. */
const buildPanels = (api: DockviewApi, environment: SchemaSessionEnvironment) => {
    const panels = new Set<string>();

    environment.cameras.forEach((camera) => {
        panels.add(camera.camera_id);
        if (!api.panels.some((panel) => panel.id === camera.camera_id)) {
            api.addPanel({
                id: camera.camera_id,
                title: camera.name,
                component: 'camera',
                params: { camera_id: camera.camera_id },
                position: { direction: 'left', referencePanel: '' },
            });
        }
    });

    panels.add(environment.robot_id);
    if (!api.panels.some((panel) => panel.id === environment.robot_id)) {
        api.addPanel({
            id: environment.robot_id,
            title: 'Follower',
            component: 'follower',
            params: { robot_id: environment.robot_id },
            position: { direction: 'right', referencePanel: '' },
        });
    }

    api.panels.filter((panel) => !panels.has(panel.id)).forEach((panel) => api.removePanel(panel));
};

const SessionDock = ({ environment }: { environment: SchemaSessionEnvironment }) => {
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent) => {
        api.current = event.api;
        buildPanels(event.api, environment);
    };

    useEffect(() => {
        if (api.current) {
            buildPanels(api.current, environment);
        }
    }, [environment]);

    return (
        <View flex backgroundColor={'gray-200'} height={'100%'} minHeight={0} position={'relative'}>
            <DockviewReact onReady={onReady} components={components} theme={physicalAiTheme} />
        </View>
    );
};

const Initializing = ({ parts, cancelHref }: { parts: SessionPart[]; cancelHref: string }) => {
    const { error } = useRuntimeSession();

    return (
        <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
            <Heading level={2}>
                <Text>Initializing</Text>
                {error === null && !parts.some((part) => part.error) && (
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                )}
            </Heading>
            <Flex direction='column' margin='size-200' gap='size-100'>
                {parts.map(({ label, isLoaded, error: partError }) => (
                    <Flex key={label} direction='column' gap='size-50'>
                        <StatusLight variant={isLoaded ? 'positive' : partError ? 'negative' : 'yellow'}>
                            {label}
                        </StatusLight>
                        {partError && <InlineAlert variant='error'>{partError}</InlineAlert>}
                    </Flex>
                ))}
                {error !== null && <InlineAlert variant='error'>{error.message}</InlineAlert>}
            </Flex>
            <Button variant={'secondary'} href={cancelHref}>
                Cancel
            </Button>
        </Flex>
    );
};

/**
 * A session's robot and cameras in a dock, under an optional toolbar.
 *
 * The middle of the session workspace. Until the environment and every one of ``parts`` is
 * loaded it shows what it is waiting for instead. Once it has shown the dock, it keeps showing
 * it while a part fails or reloads; the parts panel says what is wrong.
 */
export const SessionViewer = ({
    toolbar,
    parts,
    cancelHref,
}: {
    toolbar?: ReactNode;
    parts: SessionPart[];
    cancelHref: string;
}) => {
    const { state, statuses } = useRuntimeSession();
    const environment = state?.environment;
    const environmentFailure = failureOf(statuses, 'environment');
    const allParts = [
        {
            label: 'Environment',
            isLoaded: environment != null && environmentFailure === null,
            error: environmentFailure,
        },
        ...parts,
    ];

    const isReady = environment != null && allParts.every((part) => part.isLoaded);
    const [hasStarted, setHasStarted] = useState(false);
    // Adjusted while rendering: started once ready, and over once the environment is unloaded.
    if (hasStarted !== (isReady || (hasStarted && environment != null))) {
        setHasStarted(!hasStarted);
    }

    if (environment == null || !(isReady || hasStarted)) {
        return <Initializing parts={allParts} cancelHref={cancelHref} />;
    }

    return (
        <Flex direction={'column'} height={'100%'} gap='size-100'>
            {toolbar}
            {/* The cells suspend while they load; this keeps that inside the viewer. */}
            <Suspense fallback={<Loading mode='inline' />}>
                <SessionDock environment={environment} />
            </Suspense>
        </Flex>
    );
};
