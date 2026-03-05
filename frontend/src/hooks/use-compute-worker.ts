"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ComputeWebSocket,
  TaskMessage,
  FFNTaskMessage,
  TileTaskMessage,
  WeightsMessage,
  PipelineAssignMessage,
  PipelineForwardMessage,
  TrainingForwardMessage,
  WebRTCConnectMessage,
  WebRTCSignalMessage,
} from "@/lib/ws";
import { WebRTCManager } from "@/lib/webrtc";
import { MSG_ACTIVATIONS, type P2PMessage } from "@/lib/p2p-protocol";
import type { ShardManifest } from "@/lib/weight-store";

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
  const webrtcRef = useRef<WebRTCManager | null>(null);
  const peerIdRef = useRef<string>("");
  const manifestRef = useRef<ShardManifest[]>([]);

  const [gpuInfo, setGpuInfo] = useState<string>("");
  const [supported, setSupported] = useState(false);
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [tasksCompleted, setTasksCompleted] = useState(0);
  const [tokensEarned, setTokensEarned] = useState(0);
  const [pipelineAssignment, setPipelineAssignment] = useState<{
    startLayer: number;
    endLayer: number;
  } | null>(null);
  const [peerConnections, setPeerConnections] = useState<string[]>([]);

  const handleWorkerMessage = useCallback(
    (e: MessageEvent) => {
      const { type, payload } = e.data;

      if (type === "init_result") {
        setGpuInfo(payload.gpu);
        setSupported(payload.supported);
        if (payload.manifest) {
          manifestRef.current = payload.manifest;
        }
      }

      if (type === "manifest") {
        manifestRef.current = payload.manifest;
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
        const { layerIdx, weightsVersion } = payload;
        wsRef.current?.sendNeedWeights(layerIdx, weightsVersion);
      }

      if (type === "pipeline_result") {
        const { requestId, activations, computeTimeMs } = payload;
        // Try sending via WebRTC to downstream peer first
        const connectedPeers = webrtcRef.current?.getConnectedPeerIds() ?? [];
        let sentViaPeer = false;
        for (const peerId of connectedPeers) {
          if (webrtcRef.current?.sendActivations(peerId, requestId, activations)) {
            sentViaPeer = true;
            break;
          }
        }
        // Fallback to WebSocket relay
        if (!sentViaPeer) {
          const activationsB64 = arrayBufferToBase64(activations);
          wsRef.current?.sendPipelineResult(requestId, activationsB64, computeTimeMs);
        }
        setTasksCompleted((c) => c + 1);
      }

      if (type === "training_forward_result") {
        const { requestId, activations, computeTimeMs } = payload;
        const activationsB64 = arrayBufferToBase64(activations);
        wsRef.current?.sendTrainingForwardResult(
          requestId,
          activationsB64,
          null,
          computeTimeMs,
        );
        setTasksCompleted((c) => c + 1);
      }
    },
    [],
  );

  const initWorker = useCallback(() => {
    if (workerRef.current) return;
    const worker = new Worker(
      new URL("../workers/compute-worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.onmessage = handleWorkerMessage;
    worker.postMessage({ type: "init" });
    workerRef.current = worker;
  }, [handleWorkerMessage]);

  const start = useCallback(() => {
    if (!workerRef.current) initWorker();

    const ws = new ComputeWebSocket();
    ws.onStatusChange = setConnected;

    // Init WebRTC manager
    const webrtc = new WebRTCManager((targetPeerId, data) => {
      ws.sendWebRTCSignal(targetPeerId, data as Record<string, unknown>);
    });
    webrtc.setMessageHandler((_peerId: string, msg: P2PMessage) => {
      if (msg.type === MSG_ACTIVATIONS) {
        // Received activations from upstream peer — forward to server
        const activationsB64 = arrayBufferToBase64(msg.payload);
        ws.sendPipelineResult(msg.id, activationsB64, 0);
      }
    });
    webrtcRef.current = webrtc;

    ws.onPeerId = (msg) => {
      peerIdRef.current = msg.peer_id;
      // Send shard report after receiving peer_id
      if (manifestRef.current.length > 0) {
        ws.sendShardReport(manifestRef.current);
      }
    };

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

    ws.onPipelineAssign = (msg: PipelineAssignMessage) => {
      setPipelineAssignment({
        startLayer: msg.start_layer,
        endLayer: msg.end_layer,
      });
      // Request weights for layers we need
      for (const layerIdx of msg.weights_needed) {
        ws.sendNeedWeights(layerIdx, msg.version);
      }
    };

    ws.onPipelineForward = (msg: PipelineForwardMessage) => {
      const activations = base64ToArrayBuffer(msg.activations);
      workerRef.current?.postMessage({
        type: "pipeline_forward",
        payload: {
          requestId: msg.request_id,
          activations,
          startLayer: msg.start_layer,
          endLayer: msg.end_layer,
          seqLen: msg.seq_len,
          dModel: msg.d_model,
          startPos: msg.start_pos,
        },
      });
    };

    ws.onTrainingForward = (msg: TrainingForwardMessage) => {
      const activations = base64ToArrayBuffer(msg.activations);
      workerRef.current?.postMessage({
        type: "training_forward",
        payload: {
          requestId: msg.request_id,
          activations,
          startLayer: msg.start_layer,
          endLayer: msg.end_layer,
          seqLen: msg.seq_len,
          dModel: 512,
          dFf: 1376,
        },
      });
    };

    ws.onWebRTCConnect = (msg: WebRTCConnectMessage) => {
      webrtc.createOffer(msg.target_peer);
    };

    ws.onWebRTCSignal = (msg: WebRTCSignalMessage) => {
      if (msg.type === "webrtc_offer" && msg.sdp) {
        webrtc.handleOffer(msg.from_peer, msg.sdp);
      } else if (msg.type === "webrtc_answer" && msg.sdp) {
        webrtc.handleAnswer(msg.from_peer, msg.sdp);
      } else if (msg.type === "webrtc_ice" && msg.candidate) {
        webrtc.handleIceCandidate(msg.from_peer, msg.candidate);
      }
    };

    ws.connect();
    wsRef.current = ws;
    setRunning(true);

    // Periodically update peer connection status
    const peerInterval = setInterval(() => {
      const peers = webrtc.getConnectedPeerIds();
      setPeerConnections(peers);
    }, 2000);

    return () => clearInterval(peerInterval);
  }, [initWorker]);

  const stop = useCallback(() => {
    webrtcRef.current?.disconnectAll();
    webrtcRef.current = null;
    wsRef.current?.disconnect();
    wsRef.current = null;
    setConnected(false);
    setRunning(false);
    setPipelineAssignment(null);
    setPeerConnections([]);
  }, []);

  useEffect(() => {
    return () => {
      webrtcRef.current?.disconnectAll();
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
    pipelineAssignment,
    peerConnections,
    initWorker,
    start,
    stop,
  };
}
