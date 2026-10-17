import { ReactNode, Suspense, useState } from 'react';

import { Divider, Flex, Grid, Item, Loading, minmax, Picker, View } from '@geti-ui/ui';

import { useRuntimeSession } from '../runtime-session-provider';
import { SessionPart, SessionViewer } from '../session-viewer';
import { PanelPart, PartsPanel } from './parts-panel';

/**
 * One way of working with the session: the parts it needs, and its side bar panel, which
 * drives the follower with the mode's own control. Modes are combined deliberately, as a
 * mode of their own (human in the loop), not by showing several panels at once.
 */
export interface WorkspaceMode {
    key: string;
    label: string;
    parts: SessionPart[];
    panel: ReactNode;
}

/**
 * The session's robot and cameras in a dock, with a side bar: the session's state, the mode, and
 * the selected mode's panel. Entry points (record, and inference) only choose what is loaded
 * first and which modes there are.
 */
export const SessionWorkspace = ({
    modes,
    parts = ['dataset'],
    cancelHref,
}: {
    modes: WorkspaceMode[];
    /** What the parts panel shows besides the environment. */
    parts?: PanelPart[];
    cancelHref: string;
}) => {
    const { send } = useRuntimeSession();
    const [modeKey, setModeKey] = useState(modes[0].key);
    const mode = modes.find((candidate) => candidate.key === modeKey) ?? modes[0];

    const selectMode = (key: string) => {
        if (key === modeKey) {
            return;
        }
        // Each mode drives the follower its own way; hold it while switching.
        void send({ command: 'set_control', control: null });
        setModeKey(key);
    };

    return (
        <Grid
            columns={[minmax(0, '1fr'), 'size-4600']}
            gap='size-200'
            height='100%'
            minHeight={0}
            UNSAFE_style={{ gridTemplateRows: minmax(0, '1fr') }}
        >
            <SessionViewer parts={mode.parts} cancelHref={cancelHref} />
            <View backgroundColor='gray-100' padding='size-200' overflow='auto' minHeight={0}>
                <Flex direction='column' gap='size-200'>
                    <Suspense fallback={<Loading mode='inline' />}>
                        <PartsPanel parts={parts} />
                    </Suspense>
                    <Divider size='S' />
                    <Picker
                        label='Mode'
                        width='100%'
                        items={modes}
                        selectedKey={mode.key}
                        onSelectionChange={(key) => key !== null && selectMode(String(key))}
                    >
                        {(candidate) => <Item key={candidate.key}>{candidate.label}</Item>}
                    </Picker>
                    <Suspense fallback={<Loading mode='inline' />}>{mode.panel}</Suspense>
                </Flex>
            </View>
        </Grid>
    );
};
