import { useEffect, useRef, useState } from 'react';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ReferenceLine,
    ResponsiveContainer,
    XAxis,
    YAxis,
    type MouseHandlerDataParam,
} from 'recharts';
import { EpisodeBuffer, useRuntimeSession } from '../runtime-session-provider';


function useAnimationFrame(callback: FrameRequestCallback) {
    const callbackRef = useRef<FrameRequestCallback>(callback);
    callbackRef.current = callback;

    useEffect(() => {
        let frameId: number;

        const loop: FrameRequestCallback = (timestamp) => {
            callbackRef.current(timestamp);
            frameId = requestAnimationFrame(loop);
        };

        frameId = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(frameId);
    }, []);
}


function buildChartData(episodeBuffer: EpisodeBuffer) {
    const fps = 30;
    return episodeBuffer.actions.map((row, idx) => ({
      index: idx / fps,
      ...row
    }));
}

export default function LiveEpisodeChart() {
    const hz = 30;
    const lastFlush = useRef(0);

  const [chartData, setChartData] = useState<Record<string, number>[]>();
  const {episodeBuffer, environment} = useRuntimeSession();
  const intervalFactor = Math.exp(-episodeBuffer.current.actions.length / 100)
  const intervalMs = 1000 / (30 * intervalFactor);

useAnimationFrame((timestamp) => {
    if (timestamp - lastFlush.current >= intervalMs) {
        lastFlush.current = timestamp;
      setChartData(buildChartData(episodeBuffer.current));
    }
});
  const actions = chartData ? Object.keys(chartData[0]).filter((key) => key !== 'index') : [];
  return (
        <ResponsiveContainer width='100%' height={300} style={{ userSelect: 'none' }}>
            <LineChart
                data={chartData}
                margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
            >
                <CartesianGrid opacity={0.2} />
                <XAxis
                    dataKey='index'
                    label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -10 }}
                    //ticks={ticks}
                    type='number'
                />
                <YAxis label={{ value: 'Value (deg)', angle: -90, position: 'insideLeft' }} />

                <Legend verticalAlign='bottom' height={36} />

                {actions.map((joint, i) => (
                    <Line
                        isAnimationActive={false}
                        key={joint}
                        type='monotone'
                        dataKey={joint}
                        name={joint}
                        dot={false}
                        activeDot={false}
                        strokeWidth={2}
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
  )

}
