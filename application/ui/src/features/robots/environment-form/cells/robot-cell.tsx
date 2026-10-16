import { useState } from 'react';

import { Flex, ProgressCircle, Switch, toast, View } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../../api/errors';
import { useProjectId } from '../../../projects/use-project';
import { useRuntimeSession, useSessionJoints } from '../../../runtime-session/runtime-session-provider';
import { RobotViewer, UnavailableRobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robot-types';
import { InlineAlert } from '../../setup-wizard/shared/inline-alert';
import { useSynchronizeModelJoints } from '../../use-synchronize-model-joints';

// How often the leader is read; each reading is one command to the follower.
const TELEOP_HZ = 100;
// How long the follower takes to reach the leader when teleop starts, instead of jumping to it.
const TELEOP_GRACE_S = 3;

/** Drive the follower from the leader, or hold it where it is. */
const TeleopSwitch = () => {
    const { state, send } = useRuntimeSession();
    const [isPending, setIsPending] = useState(false);

    const setTeleop = async (teleoperate: boolean) => {
        setIsPending(true);
        const ack = await send({
            command: 'set_control',
            control: teleoperate ? { type: 'teleop', hz: TELEOP_HZ, grace_s: TELEOP_GRACE_S } : null,
        });
        setIsPending(false);
        if (!ack.ok) {
            toast.negative(ack.error?.message ?? 'Could not switch teleoperation.');
        }
    };

    return (
        <Switch
            isEmphasized
            isSelected={state?.control?.type === 'teleop'}
            isDisabled={isPending}
            onChange={(teleoperate) => void setTeleop(teleoperate)}
        >
            Teleoperate
        </Switch>
    );
};

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
                {state.environment.leader_id !== null && (
                    <View position={'absolute'} right={0} top={0} padding='size-100'>
                        <TeleopSwitch />
                    </View>
                )}
            </View>
        </RobotModelsProvider>
    );
};
