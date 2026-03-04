"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuthContext } from "@/app/providers";
import { Button } from "@/components/ui/button";

export function Nav() {
  const { user, logout } = useAuthContext();
  const pathname = usePathname();

  const navLink = (href: string, label: string) => {
    const active = pathname === href;
    return (
      <Link
        href={href}
        className={`text-sm transition-colors ${
          active
            ? "text-white font-medium"
            : "text-zinc-400 hover:text-white"
        }`}
      >
        {label}
      </Link>
    );
  };

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
              {navLink("/dashboard", "Dashboard")}
              {navLink("/compute", "Contribute")}
              {navLink("/chat", "Chat")}
              {navLink("/leaderboard", "Leaderboard")}
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <div className="flex items-center gap-1.5 rounded-full border border-amber-800/40 bg-amber-900/20 px-3 py-1">
                <span className="text-xs font-semibold text-amber-400 tabular-nums">
                  {user.token_balance}
                </span>
                <span className="text-[10px] text-amber-600">tokens</span>
              </div>
              <span className="text-sm text-zinc-500">{user.display_name}</span>
              <Button variant="ghost" size="sm" className="text-zinc-500 hover:text-white" onClick={logout}>
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
                <Button size="sm" className="bg-amber-600 hover:bg-amber-500">
                  Join the Movement
                </Button>
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
