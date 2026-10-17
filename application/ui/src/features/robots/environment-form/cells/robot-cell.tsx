import { View } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { RUNTIME_WIP_TOOLTIP } from '../../../../components/runtime-wip/runtime-wip';
import { useProjectId } from '../../../projects/use-project';
import { RobotViewer, UnavailableRobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { isUnavailableRobot } from '../../robot-types';
import { InlineAlert } from '../../setup-wizard/shared/inline-alert';

// The live preview (joint state and teleoperation) runs on the runtime session,
// which is being rebuilt. Until then the cell shows the robot's model at rest.
export const RobotCell = ({ follower_id }: { follower_id: string }) => {
    const { project_id } = useProjectId();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });
    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
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
                <RobotViewer robot={robot} />
                <View position={'absolute'} left={0} top={0} padding='size-100' maxWidth='size-4600'>
                    <InlineAlert variant='info'>Live preview: {RUNTIME_WIP_TOOLTIP}</InlineAlert>
                </View>
            </View>
        </RobotModelsProvider>
    );
};
