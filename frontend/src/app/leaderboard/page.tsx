"use client";

import { useEffect, useState } from "react";
import { api, LeaderboardResponse, TrainingStatus } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useAuthContext } from "@/app/providers";

export default function LeaderboardPage() {
  const { user } = useAuthContext();
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [training, setTraining] = useState<TrainingStatus | null>(null);

  useEffect(() => {
    api.getLeaderboard(50).then(setData).catch(() => {});
    api.getTrainingStatus().then(setTraining).catch(() => {});
    const interval = setInterval(() => {
      api.getLeaderboard(50).then(setData).catch(() => {});
      api.getTrainingStatus().then(setTraining).catch(() => {});
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-3xl font-bold">The Collective</h1>
        <p className="text-zinc-500">
          Watch the model grow in real time. Every contributor makes it smarter.
        </p>
      </div>

      {/* Model growth stats */}
      <Card className="mb-6 border-zinc-800 bg-zinc-900/50 overflow-hidden relative">
        <div className="absolute inset-0 bg-gradient-to-br from-amber-600/5 via-transparent to-emerald-600/5" />
        <CardContent className="relative pt-6">
          <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="text-center">
              <div className="text-3xl font-bold tabular-nums text-amber-400">
                {training?.current_step?.toLocaleString() ?? 0}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Training Steps</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold tabular-nums text-emerald-400">
                {training?.current_loss?.toFixed(4) ?? "N/A"}
              </div>
              <div className="mt-1 text-xs text-zinc-500">
                Model Loss <span className="text-zinc-600">(lower = smarter)</span>
              </div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold tabular-nums">
                {training?.connected_workers ?? 0}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Contributors Online</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold tabular-nums">
                v{training?.model_version ?? 1}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Model Version</div>
            </div>
          </div>

          {training && training.current_step > 0 && (
            <div>
              <div className="mb-1.5 flex justify-between text-[11px] text-zinc-500">
                <span className="flex items-center gap-1.5">
                  {training.is_training && (
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                  )}
                  {training.is_training ? "Training in progress" : "Training idle"}
                </span>
                <span>
                  {((training.current_step / 10000) * 100).toFixed(1)}% to next milestone
                </span>
              </div>
              <Progress
                value={Math.min((training.current_step / 10000) * 100, 100)}
                className="h-2"
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Collective totals */}
      {data && (
        <div className="mb-6 grid gap-4 sm:grid-cols-3">
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums">
                {data.total_contributors.toLocaleString()}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Total Contributors</div>
            </CardContent>
          </Card>
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums text-amber-400">
                {data.total_tiles.toLocaleString()}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Tiles Computed</div>
            </CardContent>
          </Card>
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums font-mono">
                {data.total_compute_time_ms > 0
                  ? `${(data.total_compute_time_ms / 1000).toFixed(1)}s`
                  : "0s"}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Total Compute Time</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Leaderboard table */}
      <Card className="border-zinc-800 bg-zinc-900/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Top Contributors</CardTitle>
        </CardHeader>
        <CardContent>
          {!data || data.top_contributors.length === 0 ? (
            <div className="py-8 text-center">
              <p className="mb-1 text-zinc-500">No contributions yet</p>
              <p className="text-sm text-zinc-600">
                Be the first to contribute compute and claim the top spot.
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              {/* Header */}
              <div className="grid grid-cols-12 px-3 py-2 text-[11px] uppercase tracking-wider text-zinc-600">
                <div className="col-span-1">#</div>
                <div className="col-span-5">Contributor</div>
                <div className="col-span-3 text-right">Tiles</div>
                <div className="col-span-3 text-right">Tokens Earned</div>
              </div>

              {data.top_contributors.map((entry) => {
                const isMe = user?.display_name === entry.display_name;
                return (
                  <div
                    key={entry.rank}
                    className={`grid grid-cols-12 items-center rounded-md px-3 py-2.5 transition-colors ${
                      isMe
                        ? "bg-amber-900/15 border border-amber-800/20"
                        : "hover:bg-zinc-800/50"
                    }`}
                  >
                    <div className="col-span-1">
                      {entry.rank <= 3 ? (
                        <span
                          className={`inline-flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${
                            entry.rank === 1
                              ? "bg-amber-500/20 text-amber-400"
                              : entry.rank === 2
                                ? "bg-zinc-400/20 text-zinc-300"
                                : "bg-orange-800/20 text-orange-400"
                          }`}
                        >
                          {entry.rank}
                        </span>
                      ) : (
                        <span className="pl-1.5 text-sm text-zinc-600">
                          {entry.rank}
                        </span>
                      )}
                    </div>
                    <div className="col-span-5">
                      <span
                        className={`text-sm font-medium ${
                          isMe ? "text-amber-400" : "text-zinc-300"
                        }`}
                      >
                        {entry.display_name}
                        {isMe && (
                          <span className="ml-2 text-[10px] text-amber-600">
                            (you)
                          </span>
                        )}
                      </span>
                    </div>
                    <div className="col-span-3 text-right">
                      <span className="text-sm tabular-nums text-zinc-400">
                        {entry.tiles_computed.toLocaleString()}
                      </span>
                    </div>
                    <div className="col-span-3 text-right">
                      <span className="text-sm font-medium tabular-nums text-emerald-400">
                        +{entry.tokens_earned.toLocaleString()}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
