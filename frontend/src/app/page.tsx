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
        <div className="relative mx-auto max-w-4xl px-4 py-28 text-center">
          <div className="mb-5 inline-block rounded-full border border-amber-800/50 bg-amber-950/40 px-4 py-1.5 text-xs font-medium tracking-wide text-amber-400 uppercase">
            AI from the people, for the people
          </div>
          <h1 className="mb-6 text-5xl font-bold tracking-tight text-white md:text-7xl">
            Own the AI
            <br />
            <span className="bg-gradient-to-r from-amber-400 via-orange-400 to-red-400 bg-clip-text text-transparent">
              you help build
            </span>
          </h1>
          <p className="mx-auto mb-10 max-w-2xl text-lg leading-relaxed text-zinc-400">
            No corporations. No data centers. Just people pooling their
            browsers to train a language model that belongs to everyone.
            Your GPU contributes a tiny piece. Together, we build something
            no single person could.
          </p>
          <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
            <Link href="/auth/register">
              <Button size="lg" className="bg-amber-600 px-8 text-white hover:bg-amber-500">
                Join the Movement
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="outline" size="lg" className="border-zinc-700">
                See Our Progress
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Manifesto strip */}
      <section className="border-b border-zinc-800 bg-zinc-900/50 py-10">
        <div className="mx-auto max-w-4xl px-4 text-center">
          <p className="text-lg font-medium text-zinc-300 md:text-xl">
            Big tech built AI behind closed doors with our data.{" "}
            <span className="text-amber-400">
              WeBrain flips the script.
            </span>{" "}
            Every browser that connects makes this model smarter, and every
            contributor earns their stake in it.
          </p>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-5xl px-4 py-20">
        <h2 className="mb-4 text-center text-3xl font-bold text-white">
          Power to the people
        </h2>
        <p className="mx-auto mb-12 max-w-xl text-center text-zinc-500">
          Three steps to join a collective that&apos;s building AI the right way
        </p>
        <div className="grid gap-6 md:grid-cols-3">
          {[
            {
              step: "01",
              title: "Claim your seat",
              desc: "Sign up and get 100 free tokens. No credit card, no data harvesting. You own your account.",
            },
            {
              step: "02",
              title: "Lend your GPU",
              desc: "Open the contribute page and hit start. Your browser computes tiny 64x64 matrix tiles \u2014 barely a whisper on your hardware.",
            },
            {
              step: "03",
              title: "Reap what we sow",
              desc: "Earn tokens for every tile. Spend them to chat with the model we\u2019re all building together. Watch it get smarter in real time.",
            },
          ].map((item) => (
            <Card
              key={item.step}
              className="border-zinc-800 bg-zinc-900/50"
            >
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
      </section>

      {/* Why this matters */}
      <section className="border-t border-zinc-800 bg-zinc-900/30 py-20">
        <div className="mx-auto max-w-4xl px-4">
          <h2 className="mb-12 text-center text-3xl font-bold text-white">
            Why this matters
          </h2>
          <div className="grid gap-8 md:grid-cols-2">
            {[
              {
                title: "No single point of control",
                desc: "The model is trained by thousands of browsers. Nobody can pull the plug, inject bias, or gate access behind a paywall.",
              },
              {
                title: "Your compute, your tokens",
                desc: "Every cycle your GPU donates earns you tokens. It\u2019s a fair exchange \u2014 the people who build it get to use it.",
              },
              {
                title: "Transparent from day one",
                desc: "Watch the loss curve drop in real time. See how many workers are connected. The whole training process is visible to everyone.",
              },
              {
                title: "Built to scale",
                desc: "The tiled architecture means the model grows with the community. More people join, bigger model we can train. Together.",
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

      {/* Live stats */}
      {stats && (
        <section className="border-t border-zinc-800 py-16">
          <div className="mx-auto max-w-4xl px-4">
            <h2 className="mb-2 text-center text-2xl font-bold text-white">
              The Collective, Live
            </h2>
            <p className="mb-8 text-center text-sm text-zinc-500">
              Real-time stats from our distributed training network
            </p>
            <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
              {[
                { label: "Training Steps", value: stats.current_step.toLocaleString() },
                { label: "Current Loss", value: stats.current_loss.toFixed(4) },
                { label: "Active Contributors", value: stats.connected_workers },
                { label: "Model Version", value: `v${stats.model_version}` },
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
            The model gets smarter with every person who joins
          </h2>
          <p className="mb-8 text-zinc-400">
            Every browser counts. Every tile matters. This is AI that doesn&apos;t
            belong to a boardroom \u2014 it belongs to you.
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
        <div className="mx-auto max-w-4xl px-4 text-center text-sm text-zinc-600">
          WeBrain \u2014 AI from the people, for the people.
        </div>
      </footer>
    </main>
  );
}
