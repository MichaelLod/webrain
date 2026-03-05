"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthContext } from "@/app/providers";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function RegisterPage() {
  const { register } = useAuthContext();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await register(email, password, displayName);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600">
            <span className="text-xl font-black text-white">W</span>
          </div>
          <h1 className="mb-1 text-2xl font-bold">Join the network</h1>
          <p className="text-sm text-zinc-500">
            Your browser becomes a node. Your GPU runs real transformer layers.
          </p>
        </div>

        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardContent className="pt-6">
            <div className="mb-6 rounded-lg border border-amber-800/30 bg-amber-900/10 px-4 py-3 text-center">
              <span className="text-sm text-amber-400">
                +100 tokens free when you join
              </span>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="rounded-lg bg-red-900/30 border border-red-800/30 p-3 text-sm text-red-400">
                  {error}
                </div>
              )}
              <div className="space-y-2">
                <Label htmlFor="name" className="text-zinc-400">Display Name</Label>
                <Input
                  id="name"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="How should we call you?"
                  required
                  className="border-zinc-700 bg-zinc-900"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email" className="text-zinc-400">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  className="border-zinc-700 bg-zinc-900"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-zinc-400">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="At least 6 characters"
                  required
                  minLength={6}
                  className="border-zinc-700 bg-zinc-900"
                />
              </div>
              <Button
                type="submit"
                className="w-full bg-amber-600 py-5 text-sm font-semibold hover:bg-amber-500"
                disabled={loading}
              >
                {loading ? "Joining..." : "Join WeBrain"}
              </Button>
              <p className="text-center text-sm text-zinc-500">
                Already have an account?{" "}
                <Link href="/auth/login" className="text-amber-400 hover:text-amber-300 transition-colors">
                  Sign in
                </Link>
              </p>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
