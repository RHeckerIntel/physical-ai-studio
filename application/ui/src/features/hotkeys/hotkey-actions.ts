export type HotkeyActionId =
    | 'recording.teleoperate'
    | 'recording.stop_teleoperation'
    | 'recording.start_episode'
    | 'recording.accept_episode'
    | 'recording.discard_episode'
    | 'relative_teleop';

type HotkeyActionDefinition = {
    label: string;
    scope: string;
    defaultCombo: string;
};

export const HOTKEY_ACTIONS: Record<HotkeyActionId, HotkeyActionDefinition> = {
    'recording.teleoperate': {
        label: 'Start teleoperation',
        scope: 'Recording',
        defaultCombo: 'ArrowRight',
    },
    'recording.stop_teleoperation': {
        label: 'Stop teleoperation',
        scope: 'Recording',
        defaultCombo: 'ArrowLeft',
    },
    'recording.start_episode': {
        label: 'Start episode',
        scope: 'Recording',
        defaultCombo: 'ArrowRight',
    },
    'recording.accept_episode': {
        label: 'Accept episode',
        scope: 'Recording',
        defaultCombo: 'ArrowRight',
    },
    'recording.discard_episode': {
        label: 'Discard episode',
        scope: 'Recording',
        defaultCombo: 'ArrowLeft',
    },
    relative_teleop: {
        label: 'Relative teleoperation (hold)',
        scope: 'Teleoperation',
        defaultCombo: 'ArrowDown',
    },
};

export const HOTKEY_ACTION_IDS = Object.keys(HOTKEY_ACTIONS) as HotkeyActionId[];
