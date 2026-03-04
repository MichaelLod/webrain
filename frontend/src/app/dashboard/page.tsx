"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthContext } from "@/app/providers";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { api, ComputeStats, TokenTransaction, TrainingStatus } from "@/lib/api";

export default function DashboardPage() {
  const { user, loading } = useAuthContext();
  const router = useRouter();
  const [stats, setStats] = useState<ComputeStats | null>(null);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [history, setHistory] = useState<TokenTransaction[]>([]);

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  useEffect(() => {
    if (!user) return;
    api.getComputeStats().then(setStats).catch(() => {});
    api.getTrainingStatus().then(setTraining).catch(() => {});
    api.getHistory(20).then(setHistory).catch(() => {});
    const interval = setInterval(() => {
      api.getTrainingStatus().then(setTraining).catch(() => {});
      api.getComputeStats().then(setStats).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, [user]);

  if (loading || !user) return null;

  const earned = stats?.tokens_earned ?? 0;
  const spent = history
    .filter((tx) => tx.amount < 0)
    .reduce((sum, tx) => sum + Math.abs(tx.amount), 0);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      {/* Welcome header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="mb-1 text-3xl font-bold">
            Hey, {user.display_name}
          </h1>
          <p className="text-zinc-500">
            Here&apos;s your contribution to the collective.
          </p>
        </div>
        <Link href="/compute">
          <Button className="bg-amber-600 hover:bg-amber-500">
            Contribute Now
          </Button>
        </Link>
      </div>

      {/* Token hero card */}
      <Card className="relative mb-6 overflow-hidden border-zinc-800 bg-zinc-900/50">
        <div className="absolute inset-0 bg-gradient-to-br from-amber-600/10 via-transparent to-transparent" />
        <CardContent className="relative grid gap-6 pt-6 sm:grid-cols-3">
          <div>
            <div className="text-sm text-zinc-500 mb-1">Your Balance</div>
            <div className="text-4xl font-bold text-amber-400 tabular-nums">
              {user.token_balance}
            </div>
            <div className="text-xs text-zinc-600 mt-1">tokens available</div>
          </div>
          <div>
            <div className="text-sm text-zinc-500 mb-1">Total Earned</div>
            <div className="text-4xl font-bold text-emerald-400 tabular-nums">
              {earned}
            </div>
            <div className="text-xs text-zinc-600 mt-1">from computing tiles</div>
          </div>
          <div>
            <div className="text-sm text-zinc-500 mb-1">Total Spent</div>
            <div className="text-4xl font-bold text-zinc-400 tabular-nums">
              {spent}
            </div>
            <div className="text-xs text-zinc-600 mt-1">on chatting with the AI</div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Your contributions */}
        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader className="pb-4">
            <CardTitle className="text-base">Your Contributions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Tiles computed</span>
              <span className="font-bold tabular-nums">
                {stats?.tasks_completed ?? 0}
              </span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Total compute time</span>
              <span className="font-mono text-sm">
                {stats
                  ? `${(stats.total_compute_time_ms / 1000).toFixed(1)}s`
                  : "0s"}
              </span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Tokens earned</span>
              <span className="font-bold text-emerald-400 tabular-nums">
                +{stats?.tokens_earned ?? 0}
              </span>
            </div>

            {(stats?.tasks_completed ?? 0) === 0 && (
              <div className="rounded-lg border border-dashed border-zinc-700 p-4 text-center">
                <p className="mb-2 text-sm text-zinc-500">
                  You haven&apos;t contributed yet
                </p>
                <Link href="/compute">
                  <Button size="sm" variant="outline" className="border-amber-800 text-amber-400 hover:bg-amber-900/30">
                    Start contributing
                  </Button>
                </Link>
              </div>
            )}
          </CardContent>
        </Card>

        {/* The Collective */}
        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">The Collective</CardTitle>
              <div className="flex items-center gap-1.5">
                <div
                  className={`h-2 w-2 rounded-full ${
                    training?.is_training
                      ? "animate-pulse bg-emerald-400"
                      : "bg-zinc-600"
                  }`}
                />
                <span className="text-xs text-zinc-500">
                  {training?.is_training ? "Training" : "Idle"}
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Active contributors</span>
              <span className="font-bold tabular-nums">
                {training?.connected_workers ?? 0}
              </span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Training steps</span>
              <span className="font-bold tabular-nums">
                {training?.current_step?.toLocaleString() ?? 0}
              </span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Model loss</span>
              <span className="font-mono text-sm text-emerald-400">
                {training?.current_loss?.toFixed(4) ?? "N/A"}
              </span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Model version</span>
              <span className="font-bold">v{training?.model_version ?? 1}</span>
            </div>

            {training && training.current_step > 0 && (
              <div>
                <div className="mb-1.5 flex justify-between text-[11px] text-zinc-600">
                  <span>Training progress</span>
                  <span>
                    {((training.current_step / 10000) * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress
                  value={Math.min(
                    (training.current_step / 10000) * 100,
                    100
                  )}
                  className="h-1.5"
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Transaction history */}
      <Card className="mt-6 border-zinc-800 bg-zinc-900/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <p className="py-4 text-center text-sm text-zinc-600">
              No activity yet. Start contributing or chatting to see your
              history.
            </p>
          ) : (
            <div className="space-y-2">
              {history.map((tx) => (
                <div
                  key={tx.id}
                  className="flex items-center justify-between rounded-md px-3 py-2 hover:bg-zinc-800/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`h-1.5 w-1.5 rounded-full ${
                        tx.amount > 0 ? "bg-emerald-400" : "bg-zinc-500"
                      }`}
                    />
                    <span className="text-sm text-zinc-300">
                      {tx.tx_type === "signup_bonus"
                        ? "Welcome bonus"
                        : tx.tx_type === "compute_reward"
                          ? "Compute reward"
                          : "Chat message"}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span
                      className={`text-sm font-medium tabular-nums ${
                        tx.amount > 0 ? "text-emerald-400" : "text-zinc-500"
                      }`}
                    >
                      {tx.amount > 0 ? "+" : ""}
                      {tx.amount}
                    </span>
                    <span className="text-[11px] text-zinc-600 tabular-nums w-12 text-right">
                      {tx.balance_after}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
