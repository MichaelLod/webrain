"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ComputeWebSocket, TaskMessage } from "@/lib/ws";

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

export function useComputeWorker() {
  const workerRef = useRef<Worker | null>(null);
  const wsRef = useRef<ComputeWebSocket | null>(null);
  const [gpuInfo, setGpuInfo] = useState<string>("");
  const [supported, setSupported] = useState(false);
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [tasksCompleted, setTasksCompleted] = useState(0);
  const [tokensEarned, setTokensEarned] = useState(0);

  const handleWorkerMessage = useCallback(
    (e: MessageEvent) => {
      const { type, payload } = e.data;

      if (type === "init_result") {
        setGpuInfo(payload.gpu);
        setSupported(payload.supported);
      }

      if (type === "result") {
        const { taskId, cTile, computeTimeMs } = payload;
        const cTileB64 = arrayBufferToBase64(cTile);
        wsRef.current?.sendResult(taskId, cTileB64, computeTimeMs);
        setTasksCompleted((c) => c + 1);
        wsRef.current?.sendReady();
      }
    },
    []
  );

  const initWorker = useCallback(() => {
    if (workerRef.current) return;
    const worker = new Worker(
      new URL("../workers/compute-worker.ts", import.meta.url),
      { type: "module" }
    );
    worker.onmessage = handleWorkerMessage;
    worker.postMessage({ type: "init" });
    workerRef.current = worker;
  }, [handleWorkerMessage]);

  const start = useCallback(() => {
    if (!workerRef.current) initWorker();

    const ws = new ComputeWebSocket();
    ws.onStatusChange = setConnected;
    ws.onTask = (task: TaskMessage) => {
      const aTile = base64ToArrayBuffer(task.a_tile);
      const bTile = base64ToArrayBuffer(task.b_tile);
      workerRef.current?.postMessage({
        type: "compute",
        payload: {
          taskId: task.task_id,
          aTile,
          bTile,
          tileSize: task.tile_size,
        },
      });
    };
    ws.onCredited = (msg) => {
      setTokensEarned((t) => t + msg.tokens_earned);
    };
    ws.connect();
    wsRef.current = ws;
    setRunning(true);
  }, [initWorker]);

  const stop = useCallback(() => {
    wsRef.current?.disconnect();
    wsRef.current = null;
    setConnected(false);
    setRunning(false);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.disconnect();
      workerRef.current?.postMessage({ type: "destroy" });
      workerRef.current?.terminate();
    };
  }, []);

  return {
    gpuInfo,
    supported,
    connected,
    running,
    tasksCompleted,
    tokensEarned,
    initWorker,
    start,
    stop,
  };
}
