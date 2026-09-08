export const WIZARD_STEPS = ['setup', 'feature-mapping', 'training-parameters', 'export'] as const;

export type WizardStep = (typeof WIZARD_STEPS)[number];

export const WIZARD_STEP_LABELS: Record<WizardStep, string> = {
    setup: 'Setup',
    'feature-mapping': 'Feature mapping',
    'training-parameters': 'Training parameters',
    export: 'Export & optimization',
};
