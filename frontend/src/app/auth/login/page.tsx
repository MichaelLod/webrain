"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthContext } from "@/app/providers";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function LoginPage() {
  const { login } = useAuthContext();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
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
          <h1 className="mb-1 text-2xl font-bold">Welcome back</h1>
          <p className="text-sm text-zinc-500">
            The network missed you.
          </p>
        </div>

        <Card className="border-zinc-800 bg-zinc-900/50">
          <CardContent className="pt-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="rounded-lg bg-red-900/30 border border-red-800/30 p-3 text-sm text-red-400">
                  {error}
                </div>
              )}
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
                  placeholder="Your password"
                  required
                  className="border-zinc-700 bg-zinc-900"
                />
              </div>
              <Button
                type="submit"
                className="w-full bg-amber-600 py-5 text-sm font-semibold hover:bg-amber-500"
                disabled={loading}
              >
                {loading ? "Signing in..." : "Sign In"}
              </Button>
              <p className="text-center text-sm text-zinc-500">
                No account?{" "}
                <Link href="/auth/register" className="text-amber-400 hover:text-amber-300 transition-colors">
                  Join the movement
                </Link>
              </p>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
