"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ComputeWebSocket,
  TaskMessage,
  FFNTaskMessage,
  TileTaskMessage,
  WeightsMessage,
} from "@/lib/ws";

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
        const { taskId, taskType, cTile, output, computeTimeMs } = payload;
        if (taskType === "ffn_forward") {
          const outputB64 = arrayBufferToBase64(output);
          wsRef.current?.sendFFNResult(taskId, outputB64, computeTimeMs);
        } else {
          const cTileB64 = arrayBufferToBase64(cTile);
          wsRef.current?.sendResult(taskId, cTileB64, computeTimeMs);
        }
        setTasksCompleted((c) => c + 1);
      }

      if (type === "need_weights") {
        // Worker needs weights it doesn't have cached — request from server
        const { layerIdx, weightsVersion } = payload;
        wsRef.current?.sendNeedWeights(layerIdx, weightsVersion);
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
      if (task.task_type === "ffn_forward") {
        const ffnTask = task as FFNTaskMessage;
        const activations = base64ToArrayBuffer(ffnTask.activations);
        const weights = ffnTask.weights
          ? {
              gate: base64ToArrayBuffer(ffnTask.weights.gate),
              up: base64ToArrayBuffer(ffnTask.weights.up),
              down: base64ToArrayBuffer(ffnTask.weights.down),
            }
          : undefined;

        workerRef.current?.postMessage({
          type: "ffn_forward",
          payload: {
            taskId: ffnTask.task_id,
            activations,
            layerIdx: ffnTask.layer_idx,
            dModel: ffnTask.d_model,
            dFf: ffnTask.d_ff,
            seqLen: ffnTask.seq_len,
            weightsVersion: ffnTask.weights_version,
            weights,
          },
        });
      } else {
        const tileTask = task as TileTaskMessage;
        const aTile = base64ToArrayBuffer(tileTask.a_tile);
        const bTile = base64ToArrayBuffer(tileTask.b_tile);
        workerRef.current?.postMessage({
          type: "compute",
          payload: {
            taskId: tileTask.task_id,
            aTile,
            bTile,
            tileSize: tileTask.tile_size,
          },
        });
      }
    };

    ws.onWeights = (msg: WeightsMessage) => {
      // Server sent weights in response to our request — cache in worker
      workerRef.current?.postMessage({
        type: "cache_weights",
        payload: {
          layerIdx: msg.layer_idx,
          weightsVersion: msg.weights_version,
          gate: base64ToArrayBuffer(msg.gate),
          up: base64ToArrayBuffer(msg.up),
          down: base64ToArrayBuffer(msg.down),
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
