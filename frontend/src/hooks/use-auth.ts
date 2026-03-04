"use client";

import { useCallback, useEffect, useState } from "react";
import { api, User } from "@/lib/api";

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchUser = useCallback(async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      setLoading(false);
      return;
    }
    try {
      const u = await api.me();
      setUser(u);
    } catch {
      localStorage.removeItem("token");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.login({ email, password });
    localStorage.setItem("token", res.token);
    setUser(res.user);
    return res.user;
  }, []);

  const register = useCallback(
    async (email: string, password: string, displayName: string) => {
      const res = await api.register({
        email,
        password,
        display_name: displayName,
      });
      localStorage.setItem("token", res.token);
      setUser(res.user);
      return res.user;
    },
    []
  );

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    setUser(null);
  }, []);

  return { user, loading, login, register, logout, refetch: fetchUser };
}
