import { StepPlaceholder } from './step-placeholder';

/**
 * Maps dataset features onto the policy's expected inputs (cameras, state, actions).
 * The form itself is not built yet — this placeholder reserves its spot in the wizard.
 */
export const FeatureMappingStep = () => (
    <StepPlaceholder
        title='Feature mapping'
        description='Mapping dataset features onto the policy inputs will be configured here.'
    />
);
