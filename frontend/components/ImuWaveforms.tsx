"use client";

import { useEffect, useRef } from "react";
import type { Frame } from "./KennelDiagram";

/** Rolling collar IMU waveforms (canvas, ~10 s window @10 Hz UI feed). */
export default function ImuWaveforms({ frame }: { frame: Frame | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const historyRef = useRef<{ ax: number[]; ay: number[]; az: number[]; gx: number[] }>({
    ax: [], ay: [], az: [], gx: [],
  });
  const MAX = 100;

  useEffect(() => {
    if (!frame) return;
    const h = historyRef.current;
    h.ax.push(frame.acc?.[0] ?? 0);
    h.ay.push(frame.acc?.[1] ?? 0);
    h.az.push(frame.acc?.[2] ?? 0);
    h.gx.push(frame.gyr?.[0] ?? 0);
    for (const key of ["ax", "ay", "az", "gx"] as const) {
      if (h[key].length > MAX) h[key].shift();
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    const series: [number[], string][] = [
      [h.ax, "#818cf8"],
      [h.ay, "#34d399"],
      [h.az, "#fbbf24"],
      [h.gx, "#f472b6"],
    ];
    const all = series.flatMap(([data]) => data);
    const min = Math.min(...all, -1);
    const max = Math.max(...all, 11);
    const pad = 4;

    series.forEach(([data, color]) => {
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (i / (MAX - 1)) * width;
        const y = height - ((v - min) / (max - min)) * (height - 2 * pad) - pad;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // legend
    ctx.font = "11px sans-serif";
    const labels = ["acc X", "acc Y", "acc Z", "gyr X"];
    series.forEach(([, color], i) => {
      ctx.fillStyle = color;
      ctx.fillRect(6 + i * 64, 6, 10, 3);
      ctx.fillText(labels[i], 20 + i * 64, 12);
    });
  }, [frame]);

  return (
    <div className="panel p-4">
      <h3 className="font-semibold mb-2">Collar IMU — micro-movements</h3>
      <canvas ref={canvasRef} width={560} height={180} className="w-full" />
    </div>
  );
}
