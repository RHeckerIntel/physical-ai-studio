import { FormEvent, useEffect, useRef, useState } from 'react';

import { Button, ButtonGroup, ComboBox, Flex, Form, Item, Keyboard, Text, toast } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaDatasetOutput } from '../../../api/openapi-spec';
import { formatKeyCombo, getEffectiveBindings } from '../../../features/hotkeys/key-combo';
import { useHotkey } from '../../../features/hotkeys/use-hotkey';
import { TELEOP_CONTROL } from '../../../features/runtime-session/cells/robot-cell';
import { useRuntimeSession } from '../../../features/runtime-session/runtime-session-provider';

import classes from './recording-toolbar.module.css';

const RecordingTimer = () => {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000);
        return () => window.clearInterval(timer);
    }, []);

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60)
        .toString()
        .padStart(2, '0');
    const remainingSeconds = (seconds % 60).toString().padStart(2, '0');

    return (
        <span className={classes.timer} aria-hidden='true'>
            {hours > 0
                ? hours.toString().padStart(2, '0') + ':' + minutes + ':' + remainingSeconds
                : minutes + ':' + remainingSeconds}
        </span>
    );
};

const ToolbarRecordingStatus = () => (
    <div className={classes.toolbarStatus} role='status' aria-live='polite'>
        <span className={classes.recordingDot} aria-hidden='true' />
        <div className={classes.statusCopy}>
            <strong>Recording episode</strong>
        </div>
        <RecordingTimer />
    </div>
);

/** Send a recording command and report a failure; ``isPending`` while one is in flight. */
const useRecordingCommand = () => {
    const { send } = useRuntimeSession();
    const [isPending, setIsPending] = useState(false);

    const run = async (command: Parameters<typeof send>[0]) => {
        setIsPending(true);
        try {
            const ack = await send(command);
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'The recording command failed.');
            }
        } finally {
            setIsPending(false);
        }
    };

    return { run, isPending };
};

/** Teleoperation is how an episode is recorded, so it starts once the dataset is loaded, if there is a leader. */
const useTeleopOnceLoaded = () => {
    const { state, send } = useRuntimeSession();
    const started = useRef(false);
    const ready = state?.dataset != null && state.environment?.leader_id != null;

    useEffect(() => {
        if (!ready || started.current) {
            return;
        }
        started.current = true;
        void send({ command: 'set_control', control: TELEOP_CONTROL });
    }, [ready, send]);
};

/** Task, start / accept / discard and the recording status, above the session viewer on the record page. */
export const RecordingToolbar = ({ dataset }: { dataset: SchemaDatasetOutput }) => {
    const { state } = useRuntimeSession();
    const [task, setTask] = useState<string>(dataset.default_task);
    const { run, isPending } = useRecordingCommand();
    useTeleopOnceLoaded();

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    const isRecording = state?.dataset?.is_recording ?? false;

    const start = () => void run({ command: 'start_episode', task });
    const save = () => void run({ command: 'save_episode' });
    const discard = () => void run({ command: 'discard_episode' });

    useHotkey(bindings['recording.start_episode'], start, !isRecording && !isPending && task !== '');
    useHotkey(bindings['recording.accept_episode'], save, isRecording && !isPending);
    useHotkey(bindings['recording.discard_episode'], discard, isRecording && !isPending);

    const onStart = (event: FormEvent) => {
        event.preventDefault();
        if (task !== '') {
            start();
        }
    };

    return (
        <Form validationBehavior='native' onSubmit={onStart}>
            <Flex justifyContent={'start'} alignItems='center' gap='size-100' height='size-800'>
                <ComboBox
                    isReadOnly={isRecording}
                    errorMessage={'A task is required in order to record.'}
                    name='Task'
                    flex
                    isRequired
                    allowsCustomValue
                    inputValue={task}
                    onInputChange={setTask}
                >
                    <Item key={dataset.default_task}>{dataset.default_task}</Item>
                </ComboBox>
                {isRecording && <ToolbarRecordingStatus />}
                {isRecording ? (
                    <ButtonGroup>
                        <Button isDisabled={isPending} variant={'negative'} onPress={discard}>
                            <Text>Discard</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>
                                {formatKeyCombo(bindings['recording.discard_episode'])}
                            </Keyboard>
                        </Button>
                        <Button isPending={isPending} onPress={save}>
                            <Text>Accept</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>
                                {formatKeyCombo(bindings['recording.accept_episode'])}
                            </Keyboard>
                        </Button>
                    </ButtonGroup>
                ) : (
                    <Button type={'submit'} isPending={isPending}>
                        <Text>Start episode</Text>
                        <Keyboard UNSAFE_className={classes.hotkey}>
                            {formatKeyCombo(bindings['recording.start_episode'])}
                        </Keyboard>
                    </Button>
                )}
            </Flex>
        </Form>
    );
};
