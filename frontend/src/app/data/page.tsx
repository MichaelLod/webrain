"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "@/app/providers";
import { api, DataSubmission, DataStats } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const CONTENT_TYPES = [
  { value: "text", label: "Article / Text" },
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
  { value: "other", label: "Other" },
];

export default function DataPage() {
  const { user, loading } = useAuthContext();
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [contentType, setContentType] = useState("text");
  const [submitting, setSubmitting] = useState(false);
  const [submissions, setSubmissions] = useState<DataSubmission[]>([]);
  const [stats, setStats] = useState<DataStats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  const refresh = useCallback(() => {
    api.getDataSubmissions(100).then((r) => setSubmissions(r.submissions)).catch(() => {});
    api.getDataStats().then(setStats).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setError("");
    setSubmitting(true);
    try {
      await api.submitData(url.trim(), contentType);
      setUrl("");
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading || !user) return null;

  const statusColor: Record<string, string> = {
    pending: "bg-zinc-700 text-zinc-300",
    fetching: "bg-amber-900/50 text-amber-400 border-amber-800",
    ready: "bg-emerald-900/50 text-emerald-400 border-emerald-800",
    failed: "bg-red-900/50 text-red-400 border-red-800",
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-3xl font-bold">Feed the Network</h1>
        <p className="text-zinc-500">
          Submit links to public content. The distributed model trains on what the community provides.
        </p>
      </div>

      {/* Data stats */}
      {stats && (
        <div className="mb-6 grid gap-4 sm:grid-cols-4">
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums">{stats.total_submissions}</div>
              <div className="mt-1 text-xs text-zinc-500">URLs Submitted</div>
            </CardContent>
          </Card>
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums text-emerald-400">{stats.ready_count}</div>
              <div className="mt-1 text-xs text-zinc-500">Ready for Training</div>
            </CardContent>
          </Card>
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums text-amber-400">
                {stats.total_text_chars > 1000000
                  ? `${(stats.total_text_chars / 1000000).toFixed(1)}M`
                  : stats.total_text_chars > 1000
                    ? `${(stats.total_text_chars / 1000).toFixed(0)}K`
                    : stats.total_text_chars}
              </div>
              <div className="mt-1 text-xs text-zinc-500">Characters Collected</div>
            </CardContent>
          </Card>
          <Card className="border-zinc-800 bg-zinc-900/50">
            <CardContent className="pt-6 text-center">
              <div className="text-2xl font-bold tabular-nums">{stats.contributors}</div>
              <div className="mt-1 text-xs text-zinc-500">Data Contributors</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Submit form */}
      <Card className="mb-6 border-zinc-800 bg-zinc-900/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Submit a URL</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="rounded-lg border border-red-800/30 bg-red-900/30 p-3 text-sm text-red-400">
                {error}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://en.wikipedia.org/wiki/..."
                required
                className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-amber-800 focus:outline-none focus:ring-1 focus:ring-amber-800/50"
              />
              <select
                value={contentType}
                onChange={(e) => setContentType(e.target.value)}
                className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2.5 text-sm text-zinc-300 focus:border-amber-800 focus:outline-none"
              >
                {CONTENT_TYPES.map((ct) => (
                  <option key={ct.value} value={ct.value}>{ct.label}</option>
                ))}
              </select>
              <Button
                type="submit"
                disabled={submitting || !url.trim()}
                className="bg-amber-600 hover:bg-amber-500"
              >
                {submitting ? "Submitting..." : "Submit"}
              </Button>
            </div>
            <p className="text-[11px] text-zinc-600">
              By submitting, you confirm this content is either publicly available or owned by you.
              Submitted data will be used to train the distributed model.
            </p>
          </form>
        </CardContent>
      </Card>

      {/* Submissions list */}
      <Card className="border-zinc-800 bg-zinc-900/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Community Submissions</CardTitle>
        </CardHeader>
        <CardContent>
          {submissions.length === 0 ? (
            <div className="py-8 text-center">
              <p className="mb-1 text-zinc-500">No data submitted yet</p>
              <p className="text-sm text-zinc-600">
                Be the first to feed the model. Paste a link to any public article, book, or text.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {submissions.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center justify-between rounded-md px-3 py-2.5 hover:bg-zinc-800/50 transition-colors"
                >
                  <div className="min-w-0 flex-1 mr-4">
                    <div className="flex items-center gap-1.5 text-sm text-zinc-300 truncate">
                      {s.content_type === "image" && (
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 text-amber-400">
                          <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                          <circle cx="8.5" cy="8.5" r="1.5" />
                          <path d="m21 15-5-5L5 21" />
                        </svg>
                      )}
                      {s.title || s.url}
                    </div>
                    {s.title && (
                      <div className="text-[11px] text-zinc-600 truncate">
                        {s.url}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {s.trained && (
                      <span className="text-[10px] text-emerald-500" title="Used in training">
                        trained
                      </span>
                    )}
                    <Badge
                      variant="secondary"
                      className={`text-[10px] ${statusColor[s.status] || ""}`}
                    >
                      {s.status}
                    </Badge>
                    <span className="text-[10px] text-zinc-600 tabular-nums w-16 text-right">
                      {s.content_type}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
