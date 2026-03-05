const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export type TileTaskMessage = {
  type: "task";
  task_type: "tile";
  task_id: string;
  a_tile: string;
  b_tile: string;
  tile_size: number;
  position: { i: number; j: number; k: number };
  meta: { step: number; layer: number; op: string };
};

export type FFNTaskMessage = {
  type: "task";
  task_type: "ffn_forward";
  task_id: string;
  activations: string;
  layer_idx: number;
  d_model: number;
  d_ff: number;
  seq_len: number;
  weights_version: string;
  weights?: {
    gate: string;
    up: string;
    down: string;
  };
};

export type WeightsMessage = {
  type: "weights";
  layer_idx: number;
  weights_version: string;
  gate: string;
  up: string;
  down: string;
  d_model: number;
  d_ff: number;
};

export type CreditedMessage = {
  type: "credited";
  tokens_earned: number;
};

export type PeerIdMessage = {
  type: "peer_id";
  peer_id: string;
};

export type ShardReportAckMessage = {
  type: "shard_report_ack";
  layers_needed: number[];
  weights_version: string;
};

export type PipelineAssignMessage = {
  type: "pipeline_assign";
  start_layer: number;
  end_layer: number;
  weights_needed: number[];
  version: string;
};

export type PipelineForwardMessage = {
  type: "pipeline_forward";
  request_id: string;
  activations: string;
  start_layer: number;
  end_layer: number;
  seq_len: number;
  d_model: number;
  start_pos: number;
};

export type TrainingForwardMessage = {
  type: "training_forward";
  request_id: string;
  activations: string;
  start_layer: number;
  end_layer: number;
  seq_len: number;
  batch_size: number;
};

export type TrainingBackwardMessage = {
  type: "training_backward";
  request_id: string;
  grad_output: string;
  saved_state: string;
  start_layer: number;
  end_layer: number;
};

export type WebRTCConnectMessage = {
  type: "webrtc_connect";
  target_peer: string;
};

export type WebRTCSignalMessage = {
  type: "webrtc_offer" | "webrtc_answer" | "webrtc_ice";
  from_peer: string;
  sdp?: string;
  candidate?: RTCIceCandidateInit;
};

export type ShardTransferRequestMessage = {
  type: "shard_transfer_request";
  layer_idx: number;
  source_peer_id: string | null;
  version: string;
};

export type TaskMessage = TileTaskMessage | FFNTaskMessage;
export type ServerMessage =
  | TaskMessage
  | WeightsMessage
  | CreditedMessage
  | PeerIdMessage
  | ShardReportAckMessage
  | PipelineAssignMessage
  | PipelineForwardMessage
  | TrainingForwardMessage
  | TrainingBackwardMessage
  | WebRTCConnectMessage
  | WebRTCSignalMessage
  | ShardTransferRequestMessage
  | { type: string };

export class ComputeWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  onTask: ((task: TaskMessage) => void) | null = null;
  onWeights: ((msg: WeightsMessage) => void) | null = null;
  onCredited: ((msg: CreditedMessage) => void) | null = null;
  onStatusChange: ((connected: boolean) => void) | null = null;
  onPeerId: ((msg: PeerIdMessage) => void) | null = null;
  onShardReportAck: ((msg: ShardReportAckMessage) => void) | null = null;
  onPipelineAssign: ((msg: PipelineAssignMessage) => void) | null = null;
  onPipelineForward: ((msg: PipelineForwardMessage) => void) | null = null;
  onTrainingForward: ((msg: TrainingForwardMessage) => void) | null = null;
  onTrainingBackward: ((msg: TrainingBackwardMessage) => void) | null = null;
  onWebRTCConnect: ((msg: WebRTCConnectMessage) => void) | null = null;
  onWebRTCSignal: ((msg: WebRTCSignalMessage) => void) | null = null;
  onShardTransferRequest: ((msg: ShardTransferRequestMessage) => void) | null = null;

  connect() {
    const token = localStorage.getItem("token");
    if (!token) return;

    this.ws = new WebSocket(`${WS_URL}/api/v1/compute/ws?token=${token}`);

    this.ws.onopen = () => {
      this.onStatusChange?.(true);
      this.sendReady();
    };

    this.ws.onmessage = (event) => {
      const data: ServerMessage = JSON.parse(event.data);
      switch (data.type) {
        case "task":
          this.onTask?.(data as TaskMessage);
          break;
        case "weights":
          this.onWeights?.(data as WeightsMessage);
          break;
        case "credited":
          this.onCredited?.(data as CreditedMessage);
          break;
        case "peer_id":
          this.onPeerId?.(data as PeerIdMessage);
          break;
        case "shard_report_ack":
          this.onShardReportAck?.(data as ShardReportAckMessage);
          break;
        case "pipeline_assign":
          this.onPipelineAssign?.(data as PipelineAssignMessage);
          break;
        case "pipeline_forward":
          this.onPipelineForward?.(data as PipelineForwardMessage);
          break;
        case "training_forward":
          this.onTrainingForward?.(data as TrainingForwardMessage);
          break;
        case "training_backward":
          this.onTrainingBackward?.(data as TrainingBackwardMessage);
          break;
        case "webrtc_connect":
          this.onWebRTCConnect?.(data as WebRTCConnectMessage);
          break;
        case "webrtc_offer":
        case "webrtc_answer":
        case "webrtc_ice":
          this.onWebRTCSignal?.(data as WebRTCSignalMessage);
          break;
        case "shard_transfer_request":
          this.onShardTransferRequest?.(data as ShardTransferRequestMessage);
          break;
      }
    };

    this.ws.onclose = () => {
      this.onStatusChange?.(false);
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private send(data: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  sendReady() {
    this.send({ type: "ready" });
  }

  sendResult(taskId: string, cTileB64: string, computeTimeMs: number) {
    this.send({
      type: "result",
      task_type: "tile",
      task_id: taskId,
      c_tile: cTileB64,
      compute_time_ms: computeTimeMs,
    });
  }

  sendFFNResult(taskId: string, outputB64: string, computeTimeMs: number) {
    this.send({
      type: "result",
      task_type: "ffn_forward",
      task_id: taskId,
      output: outputB64,
      compute_time_ms: computeTimeMs,
    });
  }

  sendNeedWeights(layerIdx: number, weightsVersion: string) {
    this.send({
      type: "need_weights",
      layer_idx: layerIdx,
      weights_version: weightsVersion,
    });
  }

  sendShardReport(manifest: { layerIdx: number; version: string; components: string[]; totalBytes: number }[]) {
    this.send({ type: "shard_report", manifest });
  }

  sendPipelineResult(requestId: string, activationsB64: string, computeTimeMs: number) {
    this.send({
      type: "pipeline_result",
      request_id: requestId,
      activations: activationsB64,
      compute_time_ms: computeTimeMs,
    });
  }

  sendPipelineReady(layersLoaded: number[]) {
    this.send({
      type: "pipeline_ready",
      layers_loaded: layersLoaded,
    });
  }

  sendTrainingForwardResult(requestId: string, activationsB64: string, savedStateB64: string | null, computeTimeMs: number) {
    this.send({
      type: "training_forward_result",
      request_id: requestId,
      activations: activationsB64,
      saved_state: savedStateB64,
      compute_time_ms: computeTimeMs,
    });
  }

  sendTrainingBackwardResult(requestId: string, gradInputB64: string, paramGradients: Record<string, string>, computeTimeMs: number) {
    this.send({
      type: "training_backward_result",
      request_id: requestId,
      grad_input: gradInputB64,
      param_gradients: paramGradients,
      compute_time_ms: computeTimeMs,
    });
  }

  sendShardTransferComplete(layerIdx: number, version: string, source: string) {
    this.send({
      type: "shard_transfer_complete",
      layer_idx: layerIdx,
      version: version,
      source: source,
    });
  }

  sendWebRTCSignal(targetPeer: string, data: Record<string, unknown>) {
    this.send({ ...data, target_peer: targetPeer });
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
