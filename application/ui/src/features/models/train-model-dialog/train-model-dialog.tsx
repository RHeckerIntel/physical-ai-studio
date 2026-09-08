import { useEffect, useMemo, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Key, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../../api/openapi-spec';
import { useProject } from '../../projects/use-project';
import { useRemoteTrainerHealth } from '../../remote-trainers/use-remote-trainer-health';
import { Stepper } from '../../robots/setup-wizard/shared/stepper';
import { ExportStep } from './export-step';
import { FeatureMappingStep } from './feature-mapping-step';
import { MODELS } from './policies';
import { SetupStep } from './setup-step';
import { TrainingDeviceInfo } from './training-device-info';
import { TrainingParameters } from './training-parameters';
import { pickBestDevice, useBestTrainingDevice } from './use-training-devices';
import { WIZARD_STEP_LABELS, WIZARD_STEPS, WizardStep } from './wizard-steps';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaJob['payload'];
};

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaJob | undefined) => void;
    defaultMaxEpochs?: number;
}

type TrainingTargetOption = {
    id: string;
    label: string;
};

export const TrainModelDialog = ({ baseModel, close, defaultMaxEpochs = 5 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');
    // Continuing an existing model needs its checkpoint, which only this machine
    // has: the trainer protocol can receive a dataset but not a base checkpoint.
    // So a resumed run offers local training only.
    const canTrainRemotely = baseModel === undefined;
    const trainingTargetOptions: TrainingTargetOption[] = [
        { id: 'local', label: 'This machine (local)' },
        ...(canTrainRemotely
            ? remoteTrainers.map((remoteTrainer) => ({
                  id: remoteTrainer.id,
                  label: remoteTrainer.name,
              }))
            : []),
    ];

    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [currentStep, setCurrentStep] = useState<WizardStep>('setup');
    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxEpochs, setMaxEpochs] = useState<number>(defaultMaxEpochs);
    const [batchSize, setBatchSize] = useState<number>(8);
    const [numWorkers, setNumWorkers] = useState<Key | null>('auto');
    const [autoScaleBatchSize, setAutoScaleBatchSize] = useState<boolean>(bestDevice?.type === 'cuda');
    const [precision, setPrecision] = useState<Key | null>(bestDevice?.type === 'cuda' ? 'bf16-mixed' : '32-true');
    const [compileModel, setCompileModel] = useState<boolean>(false);
    const [remoteTrainerId, setRemoteTrainerId] = useState<Key | null>('local');
    const isRemoteTarget = remoteTrainerId !== null && remoteTrainerId !== 'local';
    const {
        health: remoteTrainerHealth,
        isChecking: isCheckingRemoteTrainer,
        checkHealth: checkRemoteTrainerHealth,
    } = useRemoteTrainerHealth(isRemoteTarget ? (remoteTrainerId?.toString() ?? null) : null);
    const remoteUnavailable = isRemoteTarget && remoteTrainerHealth?.status === 'unreachable';
    const { data: policyAccess, isLoading: isCheckingPolicyAccess } = $api.useQuery(
        'get',
        '/api/policies/{policy}/huggingface-access',
        {
            params: { path: { policy: selectedPolicy } },
        }
    );
    const policyAccessBlocksTraining =
        isCheckingPolicyAccess ||
        policyAccess?.requirements.some(
            (requirement) =>
                requirement.required && (requirement.status === 'missing_token' || requirement.status === 'denied')
        ) === true;
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteTrainerHealth?.devices ?? []), [remoteTrainerHealth]);
    // The device actually driving this job: the local GPU when training locally,
    // or the remote trainer's reported GPU once its health check resolves. Auto
    // scale/precision defaults and the disabled state below should track whichever
    // one is currently in play, the same way they did when there was only ever a
    // single active device to consider.
    const activeDevice = isRemoteTarget ? bestRemoteDevice : bestDevice;

    useEffect(() => {
        if (activeDevice?.type === 'cuda') {
            setPrecision('bf16-mixed');
            setAutoScaleBatchSize(true);
        } else {
            setPrecision('32-true');
            setAutoScaleBatchSize(false);
        }
    }, [activeDevice]);

    const trainMutation = $api.useMutation('post', '/api/jobs:train', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });

    // Everything the job needs is picked on the setup step, so that is the only
    // step that can block progress; the later steps are free to be skipped through.
    const isSetupIncomplete =
        !selectedDataset ||
        !selectedPolicy ||
        remoteTrainerId === null ||
        remoteUnavailable ||
        policyAccessBlocksTraining;

    const currentStepIndex = WIZARD_STEPS.indexOf(currentStep);
    const isLastStep = currentStepIndex === WIZARD_STEPS.length - 1;
    const completedSteps = useMemo(() => new Set(WIZARD_STEPS.slice(0, currentStepIndex)), [currentStepIndex]);

    const goToStep = (offset: number) => {
        const nextStep = WIZARD_STEPS[currentStepIndex + offset];

        if (nextStep !== undefined) {
            setCurrentStep(nextStep);
        }
    };

    const save = async () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy || remoteTrainerId === null) {
            return;
        }

        if (isRemoteTarget) {
            // Final guard: the remote trainer may have gone offline since the last
            // poll, so re-check availability right before submitting the job.
            const latestHealth = await checkRemoteTrainerHealth();
            if (latestHealth === null || latestHealth.status === 'unreachable') {
                return;
            }
        }

        const name = baseModel?.name ?? MODELS.find((policy) => policy.id === selectedPolicy)?.name ?? '';

        const commonPayload = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy: selectedPolicy,
            max_epochs: maxEpochs,
            batch_size: batchSize,
            num_workers: numWorkers === 'auto' ? 'auto' : Number(numWorkers),
            auto_scale_batch_size: autoScaleBatchSize,
            precision: (precision?.toString() ?? 'bf16-mixed') as SchemaJob['payload']['precision'],
            compile_model: compileModel,
            val_split: 0.1,
            ...extraPayload,
        } as const;

        const payload: SchemaJob['payload'] = isRemoteTarget
            ? {
                  ...commonPayload,
                  training_target: 'remote',
                  remote_trainer_id: remoteTrainerId?.toString() ?? '',
              }
            : {
                  ...commonPayload,
                  training_target: 'local',
              };
        trainMutation.mutateAsync({ body: payload }).then((response) => {
            close(response as SchemaTrainJob | undefined);
        });
    };

    return (
        <Dialog size='L' UNSAFE_style={{ width: 'fit-content' }}>
            <Heading>
                <Flex justifyContent={'space-between'}>
                    <Text> Train model</Text>

                    <TrainingDeviceInfo
                        isRemoteTarget={isRemoteTarget}
                        remoteHealth={remoteTrainerHealth ?? null}
                        isCheckingRemote={isCheckingRemoteTrainer}
                    />
                </Flex>
            </Heading>
            <Divider />
            <Content width={'700px'}>
                <Flex direction='column' gap='size-200' width='100%'>
                    <Stepper
                        steps={[...WIZARD_STEPS]}
                        currentStep={currentStep}
                        completedSteps={completedSteps}
                        labels={WIZARD_STEP_LABELS}
                        onGoToStep={setCurrentStep}
                    />

                    <View minHeight='size-3600'>
                        {currentStep === 'setup' && (
                            <SetupStep
                                datasets={datasets}
                                selectedDataset={selectedDataset}
                                onSelectedDatasetChange={setSelectedDataset}
                                trainingTargetOptions={trainingTargetOptions}
                                remoteTrainerId={remoteTrainerId}
                                onRemoteTrainerIdChange={setRemoteTrainerId}
                                remoteUnavailable={remoteUnavailable}
                                selectedPolicy={selectedPolicy}
                                onSelectedPolicyChange={setSelectedPolicy}
                                isPolicyDisabled={baseModel !== undefined}
                                activeDevice={activeDevice}
                            />
                        )}

                        {currentStep === 'feature-mapping' && <FeatureMappingStep />}

                        {currentStep === 'training-parameters' && (
                            <TrainingParameters
                                maxEpochs={maxEpochs}
                                onMaxEpochsChange={setMaxEpochs}
                                batchSize={batchSize}
                                onBatchSizeChange={setBatchSize}
                                numWorkers={numWorkers}
                                onNumWorkersChange={setNumWorkers}
                                autoScaleBatchSize={autoScaleBatchSize}
                                onAutoScaleBatchSizeChange={setAutoScaleBatchSize}
                                precision={precision}
                                onPrecisionChange={setPrecision}
                                compileModel={compileModel}
                                onCompileModelChange={setCompileModel}
                                isAutoScaleBatchDisabled={activeDevice?.type !== 'cuda'}
                                deviceType={activeDevice?.type}
                            />
                        )}

                        {currentStep === 'export' && <ExportStep />}
                    </View>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button variant='secondary' onPress={() => goToStep(-1)} isDisabled={currentStepIndex === 0}>
                    Back
                </Button>
                {isLastStep ? (
                    <Button variant='accent' onPress={save} isDisabled={isSetupIncomplete}>
                        Train
                    </Button>
                ) : (
                    <Button variant='accent' onPress={() => goToStep(1)} isDisabled={isSetupIncomplete}>
                        Next
                    </Button>
                )}
            </ButtonGroup>
        </Dialog>
    );
};
