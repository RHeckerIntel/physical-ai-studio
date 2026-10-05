import { Divider, Flex, Heading, Item, Picker, StatusLight, Text, View, Well } from '@geti-ui/ui';

import { useEnvironmentId } from '../../features/robots/use-environment';
import { FeatureGroup, useRuntimeV2Session } from '../../features/robots/use-runtimev2-session';

const Status = ({
    isConnecting,
    error,
    environment,
    features,
}: {
    isConnecting: boolean;
    error: string | null;
    environment?: string | null;
    features?: number;
}) => {
    if (error !== null) {
        return <StatusLight variant='negative'>{error}</StatusLight>;
    }
    if (isConnecting) {
        return <StatusLight variant='notice'>Loading the environment…</StatusLight>;
    }
    return (
        <StatusLight variant='positive'>
            {environment} — {features} features
        </StatusLight>
    );
};

const JointTable = ({ group }: { group: FeatureGroup }) => (
    <View key={group.robot} marginBottom='size-300'>
        <Heading level={4}>{group.robot}</Heading>
        <Flex direction='column' gap='size-50'>
            <Flex gap='size-200'>
                <View width='size-2000'>
                    <Text>
                        <strong>joint</strong>
                    </Text>
                </View>
                <View width='size-1200'>
                    <Text>
                        <strong>observed</strong>
                    </Text>
                </View>
                <View width='size-1200'>
                    <Text>
                        <strong>commanded</strong>
                    </Text>
                </View>
            </Flex>
            {group.joints.map(({ joint, observation, action }) => (
                <Flex key={joint} gap='size-200'>
                    <View width='size-2000'>
                        <Text>{joint}</Text>
                    </View>
                    <View width='size-1200'>
                        <Text>{observation === undefined ? '—' : observation.toFixed(2)}</Text>
                    </View>
                    <View width='size-1200'>
                        <Text>{action === undefined ? '—' : action.toFixed(2)}</Text>
                    </View>
                </Flex>
            ))}
        </Flex>
    </View>
);

/**
 * A deliberately plain screen for trying the new runtime against real arms.
 *
 * The session lives as long as this page is mounted -- the robots connect on
 * arrival and are released on leaving -- so navigating away is how it ends.
 * Picking a control is what sets the arms moving; there is nothing else to
 * enable.
 */
export const EnvironmentTeleop = () => {
    const { project_id, environment_id } = useEnvironmentId();
    const { state, grouped, error, isConnecting, setControl } = useRuntimeV2Session(project_id, environment_id);

    const hasLeader = (state?.leaders.length ?? 0) > 0;
    // Exactly one thing decides the followers' commands, and choosing it is
    // what starts them moving -- there is no second switch.
    const control = state?.control?.kind ?? null;

    return (
        <View padding='size-300' overflow='auto' height='100%'>
            <Heading level={2}>Teleoperation (runtime v2)</Heading>
            <Status
                isConnecting={isConnecting}
                error={error}
                environment={state?.environment}
                features={state?.features}
            />

            <View marginTop='size-300' marginBottom='size-300'>
                <Picker
                    label='Control'
                    selectedKey={control ?? 'none'}
                    isDisabled={!hasLeader}
                    onSelectionChange={(key) => setControl(key === 'none' ? null : String(key))}
                >
                    <Item key='none'>Nothing</Item>
                    <Item key='teleop'>Leader arm</Item>
                    <Item key='model'>Policy</Item>
                </Picker>
                <Text>
                    {control === null
                        ? hasLeader
                            ? 'The arms hold where they are. Pick a control to decide their commands.'
                            : 'This environment has no leader to teleoperate from.'
                        : control === 'model'
                          ? 'A policy is deciding the commands, and the follower is following them.'
                          : 'The follower is following the leader’s position.'}
                </Text>
            </View>

            <Well marginBottom='size-300'>
                <Text>
                    The commanded column is published whether or not the switch is on, so the follower can be watched
                    before it is allowed to move. Turning the switch on sends it to the leader’s current position
                    immediately.
                </Text>
            </Well>

            <Divider size='S' marginBottom='size-200' />
            {grouped.map((group) => (
                <JointTable key={group.robot} group={group} />
            ))}
        </View>
    );
};
