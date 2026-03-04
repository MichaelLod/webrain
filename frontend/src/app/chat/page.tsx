"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "@/app/providers";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

type Message = { role: "user" | "assistant"; content: string };

const CHAT_COST = 10;

export default function ChatPage() {
  const { user, loading, refetch } = useAuthContext();
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/auth/login");
  }, [user, loading, router]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(async () => {
    if (!input.trim() || streaming) return;
    if (user && user.token_balance < CHAT_COST) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setStreaming(true);

    try {
      let assistantContent = "";
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      for await (const token of api.sendChat(userMsg, conversationId)) {
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
    }
  }, [input, streaming, user, conversationId, refetch]);

  if (loading || !user) return null;

  const canAfford = user.token_balance >= CHAT_COST;

  return (
    <div className="mx-auto flex h-[calc(100vh-3.5rem)] max-w-3xl flex-col px-4 py-4">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Talk to the People&apos;s AI</h1>
        <Badge variant="secondary" className="font-mono">
          Cost: {CHAT_COST} tokens/message
        </Badge>
      </div>

      <Card className="flex-1 overflow-y-auto border-zinc-800 bg-zinc-900/30 p-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-zinc-500">
            <div className="text-center">
              <p className="mb-2 text-lg">This AI is built by people like you</p>
              <p className="text-sm">
                Send a message and see what the collective has built so far.
                It gets smarter every time someone contributes compute.
              </p>
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`mb-4 ${msg.role === "user" ? "text-right" : ""}`}
          >
            <div
              className={`inline-block max-w-[80%] rounded-lg px-4 py-2 text-sm ${
                msg.role === "user"
                  ? "bg-violet-600 text-white"
                  : "bg-zinc-800 text-zinc-200"
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </Card>

      <div className="mt-4 flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          placeholder={canAfford ? "Type a message..." : "Insufficient tokens"}
          disabled={!canAfford || streaming}
          className="border-zinc-700 bg-zinc-900"
        />
        <Button onClick={send} disabled={!canAfford || streaming || !input.trim()}>
          {streaming ? "..." : "Send"}
        </Button>
      </div>
    </div>
  );
}
