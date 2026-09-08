import { StepPlaceholder } from './step-placeholder';

/**
 * Post-training export and optimization settings (target runtime, precision,
 * quantization). Not implemented yet — this placeholder reserves its spot in the wizard.
 */
export const ExportStep = () => (
    <StepPlaceholder
        title='Export & optimization'
        description='Export format and optimization settings for the trained model will be configured here.'
    />
);
