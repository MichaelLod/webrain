"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { api, TrainingStatus } from "@/lib/api";

export default function Landing() {
  const [stats, setStats] = useState<TrainingStatus | null>(null);

  useEffect(() => {
    api.getTrainingStatus().then(setStats).catch(() => {});
    const interval = setInterval(() => {
      api.getTrainingStatus().then(setStats).catch(() => {});
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-[calc(100vh-3.5rem)]">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-zinc-800">
        <div className="absolute inset-0 bg-gradient-to-br from-amber-950/30 via-zinc-950 to-orange-950/20" />
        {/* Animated grid lines */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: "linear-gradient(rgba(251,191,36,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(251,191,36,0.3) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }} />
        <div className="relative mx-auto max-w-4xl px-4 py-28 text-center">
          <div className="mb-5 inline-block rounded-full border border-amber-800/50 bg-amber-950/40 px-4 py-1.5 text-xs font-medium tracking-wide text-amber-400 uppercase">
            Peer-to-peer distributed AI
          </div>
          <h1 className="mb-6 text-5xl font-bold tracking-tight text-white md:text-7xl">
            The AI that
            <br />
            <span className="bg-gradient-to-r from-amber-400 via-orange-400 to-red-400 bg-clip-text text-transparent">
              lives in browsers
            </span>
          </h1>
          <p className="mx-auto mb-10 max-w-2xl text-lg leading-relaxed text-zinc-400">
            No data centers. No single point of failure. WeBrain splits a 45M-parameter
            transformer across connected browsers — your GPU runs real layers of the
            model, and activations flow peer-to-peer via WebRTC.
          </p>
          <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
            <Link href="/auth/register">
              <Button size="lg" className="bg-amber-600 px-8 text-white hover:bg-amber-500">
                Join the Network
              </Button>
            </Link>
            <Link href="/leaderboard">
              <Button variant="outline" size="lg" className="border-zinc-700">
                See the Collective
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Manifesto */}
      <section className="border-b border-zinc-800 bg-zinc-900/50 py-10">
        <div className="mx-auto max-w-4xl px-4 text-center">
          <p className="text-lg font-medium text-zinc-300 md:text-xl">
            Big tech locked AI in data centers.{" "}
            <span className="text-amber-400">
              WeBrain distributes it across the people who use it.
            </span>{" "}
            Every browser that connects adds compute. Every contributor earns
            their stake.
          </p>
        </div>
      </section>

      {/* Architecture visualization */}
      <section className="mx-auto max-w-5xl px-4 py-20">
        <h2 className="mb-4 text-center text-3xl font-bold text-white">
          A model that scales with its community
        </h2>
        <p className="mx-auto mb-12 max-w-xl text-center text-zinc-500">
          The more browsers connect, the more powerful the model becomes
        </p>

        {/* Scaling modes */}
        <div className="grid gap-4 md:grid-cols-3 mb-12">
          <Card className="border-zinc-800 bg-zinc-900/50 overflow-hidden relative group">
            <div className="absolute top-0 left-0 right-0 h-0.5 bg-zinc-700" />
            <CardContent className="pt-6">
              <div className="mb-1 flex items-center gap-2">
                <div className="flex -space-x-1">
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-amber-500" />
                </div>
                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">1 Browser</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-white">Swarm Mode</h3>
              <p className="text-sm leading-relaxed text-zinc-400">
                Each FFN layer splits into 4 expert slices. Server runs 1 locally,
                your browser runs the rest. Quality scales from 25% to 100%.
              </p>
            </CardContent>
          </Card>

          <Card className="border-zinc-800 bg-zinc-900/50 overflow-hidden relative">
            <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-amber-500 to-purple-500" />
            <CardContent className="pt-6">
              <div className="mb-1 flex items-center gap-2">
                <div className="flex -space-x-1">
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-amber-500" />
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-purple-500" />
                </div>
                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">2+ Browsers</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-white">Pipeline Mode</h3>
              <p className="text-sm leading-relaxed text-zinc-400">
                Full transformer layers assigned to each browser. Activations
                flow through the pipeline — server relays between stages.
              </p>
            </CardContent>
          </Card>

          <Card className="border-zinc-800 bg-zinc-900/50 overflow-hidden relative">
            <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-amber-500 via-purple-500 to-blue-500" />
            <CardContent className="pt-6">
              <div className="mb-1 flex items-center gap-2">
                <div className="flex -space-x-1">
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-amber-500" />
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-purple-500" />
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-blue-500" />
                  <div className="h-5 w-5 rounded-full border-2 border-zinc-900 bg-emerald-500" />
                </div>
                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">4+ Browsers</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-white">P2P Mode</h3>
              <p className="text-sm leading-relaxed text-zinc-400">
                Direct WebRTC connections between browsers. Activations flow
                peer-to-peer, bypassing the server entirely. Full redundancy.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Pipeline diagram */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-6 overflow-x-auto">
          <div className="flex items-center justify-center gap-2 min-w-[600px]">
            <div className="shrink-0 rounded-lg border border-zinc-700 bg-zinc-800/80 px-4 py-3 text-center">
              <div className="text-xs font-bold text-zinc-300">Server</div>
              <div className="text-[10px] text-zinc-500 mt-0.5">Embed tokens</div>
            </div>
            <div className="text-zinc-600 text-sm font-mono">&#8594;</div>
            {["A", "B", "C", "D"].map((label, i) => (
              <div key={label} className="flex items-center gap-2">
                <div className="shrink-0 rounded-lg border border-amber-800/40 bg-amber-900/20 px-4 py-3 text-center">
                  <div className="text-xs font-bold text-amber-400">Browser {label}</div>
                  <div className="text-[10px] text-zinc-500 mt-0.5">Layers {i * 3}-{i * 3 + 2}</div>
                </div>
                {i < 3 ? (
                  <div className="flex flex-col items-center gap-0.5">
                    <div className="text-blue-400 text-[10px] font-mono">WebRTC</div>
                    <div className="text-blue-400/60 text-sm">&#8594;</div>
                  </div>
                ) : (
                  <div className="text-zinc-600 text-sm font-mono">&#8594;</div>
                )}
              </div>
            ))}
            <div className="shrink-0 rounded-lg border border-zinc-700 bg-zinc-800/80 px-4 py-3 text-center">
              <div className="text-xs font-bold text-zinc-300">Server</div>
              <div className="text-[10px] text-zinc-500 mt-0.5">Output head</div>
            </div>
          </div>
          <div className="mt-3 text-center text-[11px] text-zinc-600">
            Activations flow through the pipeline. If WebRTC fails, the server relays.
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="border-t border-zinc-800 bg-zinc-900/30 py-20">
        <div className="mx-auto max-w-5xl px-4">
          <h2 className="mb-4 text-center text-3xl font-bold text-white">
            Three steps to join
          </h2>
          <p className="mx-auto mb-12 max-w-xl text-center text-zinc-500">
            No setup. No downloads. Just a browser with WebGPU.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            {[
              {
                step: "01",
                title: "Claim your seat",
                desc: "Sign up and get 100 free tokens. No credit card, no data harvesting. Your account, your tokens.",
              },
              {
                step: "02",
                title: "Lend your GPU",
                desc: "Hit start and your browser joins the network. It runs real transformer layers via WebGPU — attention, FFN, the works. Weights persist in IndexedDB across sessions.",
              },
              {
                step: "03",
                title: "Earn and chat",
                desc: "Earn tokens for every computation. Spend them to chat with the model the collective is building. Watch the loss drop in real time.",
              },
            ].map((item) => (
              <Card key={item.step} className="border-zinc-800 bg-zinc-900/50">
                <CardContent className="pt-6">
                  <div className="mb-3 text-3xl font-bold text-amber-500">
                    {item.step}
                  </div>
                  <h3 className="mb-2 text-lg font-semibold text-white">
                    {item.title}
                  </h3>
                  <p className="text-sm leading-relaxed text-zinc-400">{item.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Why this matters */}
      <section className="border-t border-zinc-800 py-20">
        <div className="mx-auto max-w-4xl px-4">
          <h2 className="mb-12 text-center text-3xl font-bold text-white">
            Why this matters
          </h2>
          <div className="grid gap-8 md:grid-cols-2">
            {[
              {
                title: "No single point of failure",
                desc: "The model lives across browsers. Each layer is replicated on 2+ peers. Kill a tab — the network reassigns in seconds.",
              },
              {
                title: "Your compute, your tokens",
                desc: "Every cycle your GPU donates earns you tokens. FFN layers pay 3x. Pipeline stages earn proportionally. Fair exchange.",
              },
              {
                title: "Transparent from day one",
                desc: "Watch the loss curve drop in real time. See how many peers are connected, which pipeline stages are active, and the full network topology.",
              },
              {
                title: "Built to scale",
                desc: "1 browser: expert slices. 2+: pipeline parallelism. 4+: full P2P with WebRTC. The architecture adapts to the community size.",
              },
            ].map((item) => (
              <div key={item.title} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-6">
                <h3 className="mb-2 text-base font-semibold text-amber-400">
                  {item.title}
                </h3>
                <p className="text-sm leading-relaxed text-zinc-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The model */}
      <section className="border-t border-zinc-800 bg-zinc-900/30 py-20">
        <div className="mx-auto max-w-4xl px-4">
          <h2 className="mb-4 text-center text-3xl font-bold text-white">
            WeBrainGPT
          </h2>
          <p className="mx-auto mb-10 max-w-lg text-center text-zinc-500">
            45M parameters. 12 layers. Trained entirely by browser contributors.
          </p>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { label: "Parameters", value: "45M" },
              { label: "Layers", value: "12" },
              { label: "Architecture", value: "GQA + SwiGLU + RoPE" },
              { label: "Context", value: "2,048 tokens" },
              { label: "Heads", value: "8Q / 2KV" },
              { label: "Hidden dim", value: "512" },
              { label: "FFN dim", value: "1,376" },
              { label: "Vocab", value: "8,192 BPE" },
            ].map((s) => (
              <div key={s.label} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-4 text-center">
                <div className="text-sm font-bold text-white">{s.value}</div>
                <div className="mt-1 text-[11px] text-zinc-500">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Live stats */}
      {stats && (
        <section className="border-t border-zinc-800 py-16">
          <div className="mx-auto max-w-4xl px-4">
            <h2 className="mb-2 text-center text-2xl font-bold text-white">
              The Network, Live
            </h2>
            <p className="mb-8 text-center text-sm text-zinc-500">
              Real-time stats from the distributed training network
            </p>
            <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
              {[
                { label: "Training Steps", value: stats.current_step.toLocaleString() },
                { label: "Current Loss", value: stats.current_loss.toFixed(4) },
                { label: "Active Peers", value: stats.connected_workers },
                {
                  label: "Mode",
                  value: stats.pipeline_active
                    ? `Pipeline (${stats.pipeline_stages} stages)`
                    : stats.connected_workers > 0
                      ? "Swarm"
                      : "Server",
                },
              ].map((s) => (
                <div key={s.label} className="text-center">
                  <div className="text-2xl font-bold text-white">{s.value}</div>
                  <div className="text-sm text-zinc-500">{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* CTA */}
      <section className="border-t border-zinc-800 bg-gradient-to-b from-zinc-900/50 to-zinc-950 py-20">
        <div className="mx-auto max-w-2xl px-4 text-center">
          <h2 className="mb-4 text-3xl font-bold text-white">
            Every browser makes the network stronger
          </h2>
          <p className="mb-8 text-zinc-400">
            The model gets smarter with every person who joins. Your GPU runs
            real transformer layers. This is AI that belongs to the people
            who build it.
          </p>
          <Link href="/auth/register">
            <Button size="lg" className="bg-amber-600 px-10 text-white hover:bg-amber-500">
              Join WeBrain
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-800 py-8">
        <div className="mx-auto max-w-4xl px-4 flex items-center justify-between text-sm text-zinc-600">
          <span>WeBrain &mdash; Peer-to-peer distributed AI</span>
          <a
            href="https://github.com/MichaelLod/webrain"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            GitHub
          </a>
        </div>
      </footer>
    </main>
  );
}
