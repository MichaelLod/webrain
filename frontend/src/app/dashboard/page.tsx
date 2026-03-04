"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "@/app/providers";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
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
    }, 5000);
    return () => clearInterval(interval);
  }, [user]);

  if (loading || !user) return null;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-2 text-3xl font-bold">Your Contribution</h1>
      <p className="mb-6 text-zinc-500">
        Track your impact on the collective and the AI we&apos;re building together.
      </p>

      <div className="mb-8 grid gap-4 md:grid-cols-4">
        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-400">Your Tokens</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-violet-400">
              {user.token_balance}
            </div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-400">Your Contributions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.tasks_completed ?? 0}</div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-400">Collective Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {training?.current_step?.toLocaleString() ?? 0}
            </div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-400">Model Intelligence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-emerald-400">
              {training?.current_loss?.toFixed(4) ?? "N/A"}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader>
            <CardTitle>The Collective</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">Status</span>
              <span className={training?.is_training ? "text-emerald-400" : "text-zinc-500"}>
                {training?.is_training ? "Training" : "Idle"}
              </span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">Active Contributors</span>
              <span>{training?.connected_workers ?? 0}</span>
            </div>
            <Separator className="bg-zinc-800" />
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">Model Version</span>
              <span>v{training?.model_version ?? 1}</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardHeader>
            <CardTitle>Recent Transactions</CardTitle>
          </CardHeader>
          <CardContent>
            {history.length === 0 ? (
              <p className="text-sm text-zinc-500">No transactions yet</p>
            ) : (
              <div className="space-y-2">
                {history.map((tx) => (
                  <div key={tx.id} className="flex justify-between text-sm">
                    <span className="text-zinc-400">
                      {tx.tx_type.replace("_", " ")}
                    </span>
                    <span
                      className={
                        tx.amount > 0 ? "text-emerald-400" : "text-red-400"
                      }
                    >
                      {tx.amount > 0 ? "+" : ""}
                      {tx.amount}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
