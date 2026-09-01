import { useCallback, useMemo, useRef, useState } from 'react';

import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ReferenceArea,
    ReferenceLine,
    ResponsiveContainer,
    XAxis,
    YAxis,
    type MouseHandlerDataParam,
} from 'recharts';

function buildChartData(actions: number[][], joints: string[], fps: number) {
    return actions.map((row, idx) => ({
        index: idx / fps,
        ...joints.reduce((acc, joint, index) => ({ ...acc, [joint]: row[index] }), {}),
    }));
}

enum EpisodeSourceType {
    Teleoperation,
    Model,
}

const episodeSourceTypeColors = {
    [EpisodeSourceType.Teleoperation]: '#5ac3f8',
    [EpisodeSourceType.Model]: 'red',
};

const episodeSourceTypeLabels = {
    [EpisodeSourceType.Teleoperation]: 'Teleoperation',
    [EpisodeSourceType.Model]: 'Model',
};

interface ChartSegment {
    start: number;
    end?: number;
    source: EpisodeSourceType;
}

function buildSegments(source: number[], fps: number) {
    return source.reduce<ChartSegment[]>((acc, source_point, index) => {
        if (acc.length === 0) {
            return [
                {
                    start: 0,
                    source: source_point,
                },
            ];
        }
        if (acc.at(-1)!.source !== source_point) {
            return [
                ...acc.with(-1, { ...acc.at(-1)!, end: index / fps }),
                {
                    start: index / fps,
                    source: source_point,
                },
            ];
        }
        if (index + 1 === source.length) {
            return acc.with(-1, { ...acc.at(-1)!, end: index / fps });
        }

        return acc;
    }, []);
}

const lineColors = ['#ff7300', '#387908', '#8884d8', '#82ca9d', '#ffbb28', '#a83279', '#aaffff'];

function toCapitalizedWords(name: string) {
    const words = name.match(/[A-Za-z][a-z]*/g) || [];

    return words.map(capitalize).join(' ');
}

function capitalize(word: string) {
    return word.charAt(0).toUpperCase() + word.substring(1);
}

function isPosJoint(name: string) {
    return name.endsWith('.pos');
}

function stripPosSuffix(name: string) {
    return name.replace(/\.pos$/, '');
}
interface EpisodeChartProps {
    actions: number[][];
    source: number[];
    joints: string[];
    fps: number;
    time: number;
    seek: (time: number) => void;
    play: () => void;
    pause: () => void;
    isPlaying: boolean;
}

export default function EpisodeChart({
    actions,
    source,
    joints,
    fps,
    time,
    seek,
    play,
    pause,
    isPlaying,
}: EpisodeChartProps) {
    const [hoverPosition, setHoverPosition] = useState<number | undefined>(undefined);
    const [mouseDown, setMouseDown] = useState<boolean>(false);
    const posJoints = useMemo(() => joints.filter(isPosJoint), [joints]);

    const segments = useMemo(() => {
        return buildSegments(source, fps);
    }, [source, fps]);

    const chartData = useMemo(() => {
        return buildChartData(actions, posJoints, fps);
    }, [actions, posJoints, fps]);
    const ticks = useMemo(
        () => [...Array(Math.floor((actions.length / fps) * 2)).keys()].map((m) => m / 2),
        [actions.length, fps]
    );

    console.log(segments);

    const continuePlayingOnRelease = useRef<boolean>(false);

    const handleMouseMove = useCallback(
        (nextState: MouseHandlerDataParam) => {
            if (nextState.activeLabel === undefined) {
                setHoverPosition(undefined);
            } else {
                const newTime = parseFloat(String(nextState.activeLabel ?? '0'));
                if (mouseDown) {
                    seek(newTime);
                }
                setHoverPosition(newTime);
            }
        },
        [mouseDown, seek]
    );

    const handleMouseUp = useCallback(
        (nextState: MouseHandlerDataParam) => {
            if (nextState.activeLabel !== undefined) {
                const newTime = parseFloat(String(nextState.activeLabel ?? '0'));
                seek(newTime);
            }
            setMouseDown(false);
            if (continuePlayingOnRelease.current) {
                play();
            }
        },
        [seek, play]
    );

    const handleMouseDown = useCallback(() => {
        setMouseDown(true);
        continuePlayingOnRelease.current = isPlaying;
        pause();
    }, [isPlaying, pause]);

    return (
        <ResponsiveContainer width='100%' height={300} style={{ userSelect: 'none' }}>
            <LineChart
                data={chartData}
                margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
                onMouseUp={handleMouseUp}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
            >
                <CartesianGrid opacity={0.2} />
                <XAxis
                    dataKey='index'
                    label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -10 }}
                    ticks={ticks}
                    type='number'
                />
                <YAxis label={{ value: 'Value (deg)', angle: -90, position: 'insideLeft' }} />

                <Legend verticalAlign='bottom' height={36} />
                {hoverPosition !== undefined && (
                    <ReferenceLine x={hoverPosition} stroke='#5ac3f8' label='' strokeWidth={2} />
                )}
                <ReferenceLine x={time} stroke='#ffffff' label='' strokeWidth={2} />
                {segments.map((s) => (
                    <ReferenceArea
                        key={`${s.start}-${s.end}`}
                        x1={s.start}
                        x2={s.end}
                        fill={episodeSourceTypeColors[s.source]}
                        fillOpacity={0.15}
                        ifOverflow='hidden'
                        label={{ value: episodeSourceTypeLabels[s.source], position: 'insideTop' }}
                    />
                ))}

                {posJoints.map((joint, i) => (
                    <Line
                        isAnimationActive={false}
                        key={joint}
                        type='monotone'
                        dataKey={joint}
                        stroke={lineColors.at(i % lineColors.length)}
                        name={toCapitalizedWords(stripPosSuffix(joint))}
                        dot={false}
                        activeDot={false}
                        strokeWidth={2}
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
}
