import { Content, Heading, IllustratedMessage, Text } from '@geti-ui/ui';

import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';

/**
 * The runtime that drives robots, records datasets and runs models is being
 * rebuilt on the runtime-inhousing feature branch. Features that depend on it
 * check this flag and show `RuntimeWipNotice` until their phase lands.
 * See application/docs/feature_plans/runtime-inhousing-plan.md.
 */
export const RUNTIME_WIP = true;

export const RUNTIME_WIP_TOOLTIP = 'Temporarily unavailable while the robot runtime is being reworked.';

export const RuntimeWipNotice = ({ feature }: { feature: string }) => {
    return (
        <IllustratedMessage>
            <EmptyIllustration />
            <Heading>{feature} is being reworked</Heading>
            <Content>
                <Text>The robot runtime is being rebuilt. This feature will return once that work lands.</Text>
            </Content>
        </IllustratedMessage>
    );
};
