"use client";

import Link from "next/link";
import { useAuthContext } from "@/app/providers";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export function Nav() {
  const { user, logout } = useAuthContext();

  return (
    <nav className="border-b border-zinc-800 bg-zinc-950">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 text-lg font-bold text-white">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-amber-500 to-orange-600 text-xs font-black">W</span>
            <span>WeBrain</span>
          </Link>
          {user && (
            <>
              <Link
                href="/dashboard"
                className="text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Dashboard
              </Link>
              <Link
                href="/compute"
                className="text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Contribute
              </Link>
              <Link
                href="/chat"
                className="text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Chat
              </Link>
            </>
          )}
        </div>
        <div className="flex items-center gap-4">
          {user ? (
            <>
              <Badge variant="secondary" className="font-mono">
                {user.token_balance} tokens
              </Badge>
              <span className="text-sm text-zinc-400">{user.display_name}</span>
              <Button variant="ghost" size="sm" onClick={logout}>
                Logout
              </Button>
            </>
          ) : (
            <div className="flex gap-2">
              <Link href="/auth/login">
                <Button variant="ghost" size="sm">
                  Login
                </Button>
              </Link>
              <Link href="/auth/register">
                <Button size="sm">Join the Movement</Button>
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
