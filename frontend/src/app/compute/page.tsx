"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "@/app/providers";
import { useComputeWorker } from "@/hooks/use-compute-worker";
import { api, TrainingStatus } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

export default function ComputePage() {
  const { user, loading, refetch } = useAuthContext();
  const router = useRouter();
  const {
    gpuInfo,
    supported,
    connected,
    running,
    tasksCompleted,
    tokensEarned,
    initWorker,
    start,
    stop,
  } = useComputeWorker();
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [sessionTime, setSessionTime] = useState(0);

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  useEffect(() => {
    initWorker();
  }, [initWorker]);

  useEffect(() => {
    api.getTrainingStatus().then(setTraining).catch(() => {});
    const interval = setInterval(() => {
      api.getTrainingStatus().then(setTraining).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Session timer
  useEffect(() => {
    if (!running) return;
    setSessionTime(0);
    const interval = setInterval(() => setSessionTime((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, [running]);

  // Refetch user balance when tokens change
  useEffect(() => {
    if (tokensEarned > 0) refetch();
  }, [tokensEarned, refetch]);

  if (loading || !user) return null;

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-3xl font-bold">Contribute Your Power</h1>
        <p className="text-zinc-500">
          Your GPU, our model. Every tile you compute makes the collective
          smarter.
        </p>
      </div>

      {/* Main control area */}
      <Card className="relative mb-6 overflow-hidden border-zinc-800 bg-zinc-900/50">
        {running && (
          <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 via-transparent to-emerald-500/5" />
        )}
        <CardContent className="relative pt-6">
          {/* GPU info row */}
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`h-3 w-3 rounded-full ${
                  running
                    ? "animate-pulse bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                    : supported
                      ? "bg-zinc-600"
                      : "bg-red-500"
                }`}
              />
              <div>
                <div className="text-sm font-medium">
                  {running ? "Contributing" : supported ? "Ready" : "Unavailable"}
                </div>
                <div className="text-xs text-zinc-500 font-mono">
                  {gpuInfo || "Detecting GPU..."}
                </div>
              </div>
            </div>
            <Badge
              variant={connected ? "default" : "secondary"}
              className={connected ? "bg-emerald-900/50 text-emerald-400 border-emerald-800" : ""}
            >
              {connected ? "Connected" : "Offline"}
            </Badge>
          </div>

          {/* Big button */}
          {!running ? (
            <Button
              onClick={start}
              disabled={!supported}
              size="lg"
              className="w-full bg-amber-600 py-6 text-lg font-semibold hover:bg-amber-500 disabled:opacity-40"
            >
              {!supported ? "WebGPU Not Available" : "Start Contributing"}
            </Button>
          ) : (
            <div className="space-y-6">
              {/* Live activity visualization */}
              <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-6">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="relative flex h-5 w-5 items-center justify-center">
                      <div className="absolute h-5 w-5 animate-ping rounded-full bg-amber-500/30" />
                      <div className="h-2.5 w-2.5 rounded-full bg-amber-500" />
                    </div>
                    <span className="text-sm font-medium text-amber-400">
                      Your GPU is powering the people&apos;s AI
                    </span>
                  </div>
                  <span className="font-mono text-xs text-zinc-500">
                    {formatTime(sessionTime)}
                  </span>
                </div>

                {/* Stats grid */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="rounded-lg bg-zinc-900/50 p-3 text-center">
                    <div className="text-2xl font-bold tabular-nums">
                      {tasksCompleted}
                    </div>
                    <div className="text-[11px] text-zinc-500 uppercase tracking-wider">
                      Tiles
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-900/50 p-3 text-center">
                    <div className="text-2xl font-bold tabular-nums text-amber-400">
                      +{tokensEarned}
                    </div>
                    <div className="text-[11px] text-zinc-500 uppercase tracking-wider">
                      Tokens
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-900/50 p-3 text-center">
                    <div className="text-2xl font-bold tabular-nums text-emerald-400">
                      {user.token_balance}
                    </div>
                    <div className="text-[11px] text-zinc-500 uppercase tracking-wider">
                      Balance
                    </div>
                  </div>
                </div>
              </div>

              <Button
                onClick={stop}
                variant="outline"
                className="w-full border-zinc-700"
              >
                Stop Contributing
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Collective status */}
      <Card className="mb-6 border-zinc-800 bg-zinc-900/50">
        <CardContent className="pt-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
              The Collective
            </h3>
            {training?.is_training && (
              <Badge variant="secondary" className="bg-emerald-900/30 text-emerald-400 border-emerald-800 text-[10px]">
                Training Active
              </Badge>
            )}
          </div>

          {/* Collective Intelligence meter */}
          <div className="mb-5 rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                Collective Intelligence
              </span>
              <span className="text-sm font-bold tabular-nums text-amber-400">
                {training
                  ? `${Math.round((training.collective_intelligence ?? 0.25) * 100)}%`
                  : "25%"}
              </span>
            </div>
            <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-amber-600 to-amber-400 transition-all duration-700"
                style={{
                  width: `${Math.round((training?.collective_intelligence ?? 0.25) * 100)}%`,
                }}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
              <span>
                {training?.active_experts ?? 1} / 4 experts active
              </span>
              <span>
                {(training?.connected_workers ?? 0) === 0
                  ? "Server only"
                  : `${training?.connected_workers} browser${(training?.connected_workers ?? 0) > 1 ? "s" : ""} contributing`}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div>
              <div className="text-lg font-bold tabular-nums">
                {training?.connected_workers ?? 0}
              </div>
              <div className="text-xs text-zinc-500">Contributors online</div>
            </div>
            <div>
              <div className="text-lg font-bold tabular-nums">
                {training?.current_step?.toLocaleString() ?? 0}
              </div>
              <div className="text-xs text-zinc-500">Training steps</div>
            </div>
            <div>
              <div className="text-lg font-bold tabular-nums text-emerald-400">
                {training?.current_loss?.toFixed(4) ?? "N/A"}
              </div>
              <div className="text-xs text-zinc-500">Loss (lower = smarter)</div>
            </div>
            <div>
              <div className="text-lg font-bold tabular-nums">
                v{training?.model_version ?? 1}
              </div>
              <div className="text-xs text-zinc-500">Model version</div>
            </div>
          </div>
          {training && training.current_step > 0 && (
            <div className="mt-4">
              <div className="mb-1 flex justify-between text-[11px] text-zinc-500">
                <span>Training progress</span>
                <span>{training.current_step.toLocaleString()} steps</span>
              </div>
              <Progress
                value={Math.min((training.current_step / 10000) * 100, 100)}
                className="h-1.5"
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* How it works - collapsed */}
      <details className="group rounded-lg border border-zinc-800 bg-zinc-900/50">
        <summary className="cursor-pointer px-6 py-4 text-sm font-medium text-zinc-400 hover:text-white transition-colors">
          How your contribution works
        </summary>
        <div className="border-t border-zinc-800 px-6 py-4">
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              {
                icon: "01",
                text: "Each FFN layer is split into 4 expert slices \u2014 the server always runs 1 locally (25% baseline)",
              },
              {
                icon: "02",
                text: "Your browser computes additional expert slices using WebGPU \u2014 more browsers = smarter model",
              },
              {
                icon: "03",
                text: "Expert outputs are summed \u2014 with all 4 active, the result is mathematically identical to the full model",
              },
              {
                icon: "04",
                text: "You earn tokens for every expert slice computed \u2014 fair exchange for collective intelligence",
              },
            ].map((step) => (
              <div key={step.icon} className="flex gap-3 text-sm">
                <span className="shrink-0 font-bold text-amber-500/60">
                  {step.icon}
                </span>
                <span className="text-zinc-400">{step.text}</span>
              </div>
            ))}
          </div>
        </div>
      </details>
    </div>
  );
}
