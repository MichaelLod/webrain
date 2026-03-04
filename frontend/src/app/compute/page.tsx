"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "@/app/providers";
import { useComputeWorker } from "@/hooks/use-compute-worker";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

export default function ComputePage() {
  const { user, loading } = useAuthContext();
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

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  useEffect(() => {
    initWorker();
  }, [initWorker]);

  if (loading || !user) return null;

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="mb-2 text-3xl font-bold">Contribute Your Power</h1>
      <p className="mb-6 text-zinc-500">
        Every tile you compute pushes our collective AI forward. Your GPU, our model.
      </p>

      <Card className="mb-6 border-zinc-800 bg-zinc-900/50">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            GPU Status
            <Badge variant={supported ? "default" : "destructive"}>
              {supported ? "WebGPU Ready" : "Not Supported"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">GPU</span>
            <span className="font-mono">{gpuInfo || "Detecting..."}</span>
          </div>
          <Separator className="bg-zinc-800" />
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">Connection</span>
            <span className={connected ? "text-emerald-400" : "text-zinc-500"}>
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6 border-zinc-800 bg-zinc-900/50">
        <CardHeader>
          <CardTitle>Worker Control</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <Button
              onClick={start}
              disabled={!supported || running}
              className="flex-1"
            >
              {running ? "Contributing..." : "Start Contributing"}
            </Button>
            <Button
              onClick={stop}
              variant="outline"
              disabled={!running}
              className="flex-1"
            >
              Stop
            </Button>
          </div>

          {running && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
              <div className="mb-2 flex items-center gap-2">
                <div className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                <span className="text-sm text-emerald-400">
                  Your GPU is powering the people&apos;s AI...
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-zinc-400">Tasks Completed</span>
                  <div className="text-xl font-bold">{tasksCompleted}</div>
                </div>
                <div>
                  <span className="text-zinc-400">Tokens Earned</span>
                  <div className="text-xl font-bold text-violet-400">
                    +{tokensEarned}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-zinc-800 bg-zinc-900/50">
        <CardHeader>
          <CardTitle>How your contribution works</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="list-inside list-decimal space-y-2 text-sm text-zinc-400">
            <li>
              Training gets broken into tiny 64x64 matrix tiles that any device can handle
            </li>
            <li>
              Your browser picks up a tile and runs it on your GPU via WebGPU
            </li>
            <li>
              The result flows back and gets assembled with everyone else&apos;s work
            </li>
            <li>You earn tokens for every tile &mdash; fair exchange for your contribution</li>
            <li>
              Together, thousands of browsers train one shared AI
            </li>
          </ol>
        </CardContent>
      </Card>
    </div>
  );
}
