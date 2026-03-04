const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  return res.json();
}

export const api = {
  register: (data: { email: string; password: string; display_name: string }) =>
    apiFetch<{ token: string; user: User }>("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  login: (data: { email: string; password: string }) =>
    apiFetch<{ token: string; user: User }>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  me: () => apiFetch<User>("/api/v1/auth/me"),

  getBalance: () => apiFetch<{ balance: number }>("/api/v1/tokens/balance"),

  getHistory: (limit = 50) =>
    apiFetch<TokenTransaction[]>(`/api/v1/tokens/history?limit=${limit}`),

  getComputeStats: () => apiFetch<ComputeStats>("/api/v1/compute/stats"),

  getTrainingStatus: () =>
    apiFetch<TrainingStatus>("/api/v1/compute/training-status"),

  getLeaderboard: (limit = 20) =>
    apiFetch<LeaderboardResponse>(`/api/v1/leaderboard?limit=${limit}`),

  submitData: (url: string, content_type = "text") =>
    apiFetch<DataSubmission>("/api/v1/data/submit", {
      method: "POST",
      body: JSON.stringify({ url, content_type }),
    }),

  getDataSubmissions: (limit = 50) =>
    apiFetch<{ submissions: DataSubmission[]; total: number }>(
      `/api/v1/data/submissions?limit=${limit}`
    ),

  getDataStats: () => apiFetch<DataStats>("/api/v1/data/stats"),

  sendChat: async function* (message: string, conversationId?: string) {
    const token = localStorage.getItem("token");
    const res = await fetch(`${API_URL}/api/v1/chat/send`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ message, conversation_id: conversationId }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Chat error ${res.status}`);
    }
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;
          yield data;
        }
      }
    }
  },
};

export interface User {
  id: number;
  email: string;
  display_name: string;
  token_balance: number;
  compute_trust_score?: number;
}

export interface TokenTransaction {
  id: number;
  amount: number;
  tx_type: string;
  reference_id: string | null;
  balance_after: number;
  created_at: string;
}

export interface ComputeStats {
  tasks_completed: number;
  total_compute_time_ms: number;
  tokens_earned: number;
}

export interface LeaderboardEntry {
  rank: number;
  display_name: string;
  tiles_computed: number;
  tokens_earned: number;
}

export interface LeaderboardResponse {
  top_contributors: LeaderboardEntry[];
  total_contributors: number;
  total_tiles: number;
  total_compute_time_ms: number;
}

export interface DataSubmission {
  id: number;
  url: string;
  content_type: string;
  status: string;
  title: string | null;
  created_at: string;
}

export interface DataStats {
  total_submissions: number;
  ready_count: number;
  total_text_chars: number;
  contributors: number;
}

export interface TrainingStatus {
  current_step: number;
  current_loss: number;
  total_flops: number;
  model_version: number;
  connected_workers: number;
  is_training: boolean;
}
