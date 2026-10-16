import { Flex, ProgressCircle, View } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../../api/errors';
import { useProjectId } from '../../../projects/use-project';
import { useRuntimeSession, useSessionJoints } from '../../../runtime-session/runtime-session-provider';
import { RobotViewer, UnavailableRobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robot-types';
import { InlineAlert } from '../../setup-wizard/shared/inline-alert';
import { useSynchronizeModelJoints } from '../../use-synchronize-model-joints';

const LiveRobot = ({ robot }: { robot: AvailableSchemaRobot }) => {
    const joints = useSessionJoints();
    useSynchronizeModelJoints(joints, robot.type);

    return <RobotViewer robot={robot} />;
};

export const RobotCell = ({ follower_id }: { follower_id: string }) => {
    const { project_id } = useProjectId();
    const { state, error, isLoading } = useRuntimeSession();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });
    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
    }

    if (error !== null && !isLoading) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <InlineAlert variant='error'>
                    <strong>{getRobotConnectionErrorTitle(error.error_code)}</strong>
                    <br />
                    {error.message}
                </InlineAlert>
            </Flex>
        );
    }

    if (state?.environment?.robot_id !== follower_id) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Connecting robot' />
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <View
                minWidth='size-4000'
                minHeight='size-4000'
                width='100%'
                height='100%'
                backgroundColor={'gray-600'}
                position={'relative'}
            >
                <LiveRobot robot={robot} />
            </View>
        </RobotModelsProvider>
    );
};
