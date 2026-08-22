"use client";

import React from "react";
import { Activity, Dna, FlaskConical } from "lucide-react";

export interface Biomarker {
  compound: string;
  importance_score: number;
  clinical_impact: string;
}

interface BiomarkerWaterfallProps {
  biomarkers: Biomarker[];
  pathways?: Record<string, number>;
}

export default function BiomarkerWaterfall({
  biomarkers,
  pathways,
}: BiomarkerWaterfallProps) {
  if (!biomarkers || biomarkers.length === 0) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        No biomarker attribution computed for this window.
      </div>
    );
  }

  const maxScore = Math.max(...biomarkers.map((b) => b.importance_score), 0.1);

  return (
    <div className="space-y-4">
      {/* Top Attributed VOCs */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span className="flex items-center gap-1.5 font-semibold text-gray-300">
            <FlaskConical className="w-3.5 h-3.5 text-indigo-400" />
            Top Attributed VOC Metabolites (Reverse Hilbert Mapping)
          </span>
          <span>Importance Score</span>
        </div>

        <div className="space-y-2">
          {biomarkers.map((b, idx) => {
            const isElevated = b.clinical_impact.includes("Elevated") || b.clinical_impact.includes("Risk");
            const widthPct = Math.min(100, Math.max(8, (b.importance_score / maxScore) * 100));

            return (
              <div key={idx} className="bg-[#0b0f19] border border-white/5 rounded-lg p-2.5 space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-200">{b.compound}</span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                        isElevated
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      }`}
                    >
                      {b.clinical_impact}
                    </span>
                  </div>
                  <span className="font-mono text-purple-300 font-bold">
                    {b.importance_score.toFixed(4)}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isElevated
                        ? "bg-gradient-to-r from-rose-500 to-purple-600"
                        : "bg-gradient-to-r from-emerald-500 to-teal-400"
                    }`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Biological Pathway Breakdown */}
      {pathways && Object.keys(pathways).length > 0 && (
        <div className="space-y-2 pt-2 border-t border-white/5">
          <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
            <Dna className="w-3.5 h-3.5 text-purple-400" />
            Biochemical Pathway Dysregulation
          </span>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(pathways).map(([pathway, pct], i) => (
              <div key={i} className="p-2.5 bg-[#0b0f19] border border-white/5 rounded-lg flex flex-col justify-between">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider line-clamp-1">
                  {pathway.replace(/_/g, " ")}
                </span>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-sm font-bold text-indigo-300 font-mono">{pct}%</span>
                  <div className="w-12 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
