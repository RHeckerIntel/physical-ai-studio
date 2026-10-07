export type HotkeyActionId = 'recording.start_episode' | 'recording.accept_episode' | 'recording.discard_episode' | 'recording.toggle_teleoperation';

type HotkeyActionDefinition = {
    label: string;
    scope: string;
    defaultCombo: string;
};

export const HOTKEY_ACTIONS: Record<HotkeyActionId, HotkeyActionDefinition> = {
    'recording.toggle_teleoperation': {
        label: 'Toggle Teleoperation',
        scope: 'Recording',
        defaultCombo: 'ArrowUp',
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
};

export const HOTKEY_ACTION_IDS = Object.keys(HOTKEY_ACTIONS) as HotkeyActionId[];
