import { ReactNode, Suspense, useEffect, useRef } from 'react';

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
import { useRuntimeSession } from './runtime-session-provider';

/** Something a page waits for before its session is usable, such as the environment or a dataset. */
export interface SessionPart {
    label: string;
    isLoaded: boolean;
}

const components = {
    follower: (props: IDockviewPanelProps<{ robot_id: string }>) => <RobotCell follower_id={props.params.robot_id} />,
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
                {error === null && (
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                )}
            </Heading>
            <Flex direction='column' margin='size-200' gap='size-100'>
                {parts.map(({ label, isLoaded }) => (
                    <StatusLight key={label} variant={isLoaded ? 'positive' : 'yellow'}>
                        {label}
                    </StatusLight>
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
 * A session's robot and cameras in a dock, under the page's own toolbar.
 *
 * Shared by the pages that run a session on a robot (recording, and inference). Until the
 * environment and every one of ``parts`` is loaded it shows what it is waiting for instead.
 */
export const SessionViewer = ({
    toolbar,
    parts,
    cancelHref,
}: {
    toolbar: ReactNode;
    parts: SessionPart[];
    cancelHref: string;
}) => {
    const { state } = useRuntimeSession();
    const environment = state?.environment;
    const allParts = [{ label: 'Environment', isLoaded: environment != null }, ...parts];

    if (environment == null || allParts.some((part) => !part.isLoaded)) {
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
