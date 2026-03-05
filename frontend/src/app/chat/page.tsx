"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthContext } from "@/app/providers";
import { api, TrainingStatus } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Message = { role: "user" | "assistant"; content: string; imageUrl?: string };

const CHAT_COST = 10;
const CHAT_IMAGE_COST = 20;

export default function ChatPage() {
  const { user, loading, refetch } = useAuthContext();
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    api.getTrainingStatus().then(setTraining).catch(() => {});
  }, []);

  // Auto-resize textarea
  const adjustTextarea = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, []);

  const currentCost = selectedImage ? CHAT_IMAGE_COST : CHAT_COST;

  const send = useCallback(async () => {
    if (!input.trim() || streaming) return;
    if (user && user.token_balance < currentCost) return;

    const userMsg = input.trim();
    const image = selectedImage;
    const preview = imagePreview;
    setInput("");
    setSelectedImage(null);
    setImagePreview(null);
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }
    setMessages((prev) => [...prev, { role: "user", content: userMsg, imageUrl: preview || undefined }]);
    setStreaming(true);

    try {
      let assistantContent = "";
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      const stream = image
        ? api.sendChatWithImage(userMsg, image, conversationId)
        : api.sendChat(userMsg, conversationId);

      for await (const token of stream) {
        assistantContent += token;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: assistantContent,
          };
          return updated;
        });
      }
      refetch();
    } catch (err) {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : "Failed to get response"}`,
        },
      ]);
    } finally {
      setStreaming(false);
      inputRef.current?.focus();
    }
  }, [input, streaming, user, conversationId, refetch, selectedImage, imagePreview, currentCost]);

  if (loading || !user) return null;

  const canAfford = user.token_balance >= currentCost;
  const messageCount = messages.filter((m) => m.role === "user").length;

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col px-4 py-4">
      {/* Header bar */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="relative flex h-5 w-5 items-center justify-center">
              {training?.is_training && (
                <div className="absolute h-5 w-5 animate-ping rounded-full bg-emerald-500/20" />
              )}
              <div
                className={`h-2.5 w-2.5 rounded-full ${
                  training?.is_training
                    ? "bg-emerald-400"
                    : "bg-zinc-600"
                }`}
              />
            </div>
            <h1 className="text-lg font-semibold">The People&apos;s AI</h1>
          </div>
          {training && (
            <span className="text-xs text-zinc-500 font-mono">
              v{training.model_version} · step {training.current_step.toLocaleString()}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant="secondary"
            className="font-mono text-xs bg-zinc-800 text-zinc-400 border-zinc-700"
          >
            {currentCost} tokens/msg
          </Badge>
          <Badge
            variant="secondary"
            className={`font-mono text-xs ${
              canAfford
                ? "bg-amber-900/30 text-amber-400 border-amber-800"
                : "bg-red-900/30 text-red-400 border-red-800"
            }`}
          >
            {user.token_balance} left
          </Badge>
        </div>
      </div>

      {/* Chat area */}
      <Card className="relative flex-1 overflow-hidden border-zinc-800 bg-zinc-900/30">
        <div className="absolute inset-0 overflow-y-auto p-4 pb-2">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center">
              <div className="mb-8 text-center max-w-md">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500/20 to-orange-600/20 border border-amber-800/30">
                  <span className="text-2xl font-black text-amber-400">W</span>
                </div>
                <h2 className="mb-2 text-xl font-semibold text-zinc-200">
                  Talk to the collective
                </h2>
                <p className="mb-4 text-sm text-zinc-500 leading-relaxed">
                  This model is distributed across browser nodes. Each peer runs real
                  transformer layers via WebGPU. It gets smarter with every node that joins.
                </p>
                {training && (
                  <div className="inline-flex items-center gap-4 rounded-lg border border-zinc-800 bg-zinc-900/80 px-4 py-2 text-xs text-zinc-500">
                    <span>
                      <span className="text-zinc-400">{training.connected_workers}</span> nodes
                    </span>
                    <span className="text-zinc-700">·</span>
                    <span>
                      <span className="text-zinc-400">{training.current_step.toLocaleString()}</span> steps trained
                    </span>
                    <span className="text-zinc-700">·</span>
                    <span>
                      loss <span className="text-emerald-400 font-mono">{training.current_loss?.toFixed(4) ?? "N/A"}</span>
                    </span>
                  </div>
                )}
              </div>

              <div className="grid w-full max-w-md gap-2">
                {[
                  "What can you tell me about yourself?",
                  "Write a short poem",
                  "Explain how you were trained",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => {
                      setInput(suggestion);
                      inputRef.current?.focus();
                    }}
                    disabled={!canAfford}
                    className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-2.5 text-left text-sm text-zinc-400 transition-colors hover:border-zinc-700 hover:bg-zinc-800/50 hover:text-zinc-300 disabled:opacity-40"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.role === "assistant" && (
                    <div className="mr-2 mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gradient-to-br from-amber-500/20 to-orange-600/20 border border-amber-800/30">
                      <span className="text-[10px] font-black text-amber-400">W</span>
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-amber-600 text-white rounded-br-md"
                        : "bg-zinc-800/80 text-zinc-200 rounded-bl-md"
                    }`}
                  >
                    {msg.imageUrl && (
                      <img
                        src={msg.imageUrl}
                        alt="Uploaded"
                        className="mb-2 max-h-48 rounded-lg object-cover"
                      />
                    )}
                    <pre className="whitespace-pre-wrap font-sans">
                      {msg.content}
                      {msg.role === "assistant" &&
                        i === messages.length - 1 &&
                        streaming && (
                          <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-amber-400" />
                        )}
                    </pre>
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </Card>

      {/* Input area */}
      <div className="mt-3">
        {!canAfford ? (
          <div className="rounded-lg border border-dashed border-zinc-700 bg-zinc-900/50 p-4 text-center">
            <p className="mb-2 text-sm text-zinc-500">
              You need {currentCost} tokens per message.
              Your balance: <span className="text-red-400 font-mono">{user.token_balance}</span>
            </p>
            <Link href="/compute">
              <Button
                size="sm"
                variant="outline"
                className="border-amber-800 text-amber-400 hover:bg-amber-900/30"
              >
                Earn tokens by running a node
              </Button>
            </Link>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {imagePreview && (
              <div className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900/80 px-3 py-2">
                <img src={imagePreview} alt="Preview" className="h-12 w-12 rounded object-cover" />
                <span className="text-xs text-zinc-400 truncate flex-1">{selectedImage?.name}</span>
                <button
                  onClick={() => { setSelectedImage(null); setImagePreview(null); }}
                  className="text-zinc-500 hover:text-zinc-300 text-sm"
                >
                  &times;
                </button>
              </div>
            )}
            <div className="flex gap-2 items-center">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setSelectedImage(file);
                    setImagePreview(URL.createObjectURL(file));
                  }
                  e.target.value = "";
                }}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={streaming}
                className="flex h-[42px] w-10 shrink-0 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300 transition-colors disabled:opacity-50"
                title="Attach image"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="m21 15-5-5L5 21" />
                </svg>
              </button>
              <div className="relative flex-1">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    adjustTextarea();
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                  placeholder={selectedImage ? "Describe this image..." : "Type a message..."}
                  disabled={streaming}
                  rows={1}
                  className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 pr-20 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-amber-800 focus:outline-none focus:ring-1 focus:ring-amber-800/50 disabled:opacity-50"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2">
                  <Button
                    onClick={send}
                    disabled={streaming || !input.trim()}
                    size="sm"
                    className="h-8 bg-amber-600 px-3 text-xs hover:bg-amber-500 disabled:opacity-30"
                  >
                    {streaming ? (
                      <span className="flex items-center gap-1.5">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" />
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white [animation-delay:150ms]" />
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white [animation-delay:300ms]" />
                      </span>
                    ) : (
                      "Send"
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
        {messages.length > 0 && canAfford && (
          <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-600">
            <span>
              {messageCount} message{messageCount !== 1 ? "s" : ""} sent · {messageCount * CHAT_COST} tokens spent this session
            </span>
            <button
              onClick={() => {
                setMessages([]);
                setConversationId(undefined);
              }}
              className="text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              Clear chat
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
