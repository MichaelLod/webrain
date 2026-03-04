const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export type TileTaskMessage = {
  type: "task";
  task_type: "tile";
  task_id: string;
  a_tile: string; // base64
  b_tile: string; // base64
  tile_size: number;
  position: { i: number; j: number; k: number };
  meta: { step: number; layer: number; op: string };
};

export type FFNTaskMessage = {
  type: "task";
  task_type: "ffn_forward";
  task_id: string;
  activations: string; // base64 float32
  layer_idx: number;
  d_model: number;
  d_ff: number;
  seq_len: number;
  weights_version: string;
  weights?: {
    gate: string; // base64 float32, transposed [D, D_ff]
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

export type TaskMessage = TileTaskMessage | FFNTaskMessage;
export type ServerMessage = TaskMessage | WeightsMessage | CreditedMessage | { type: string };

export class ComputeWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  onTask: ((task: TaskMessage) => void) | null = null;
  onWeights: ((msg: WeightsMessage) => void) | null = null;
  onCredited: ((msg: CreditedMessage) => void) | null = null;
  onStatusChange: ((connected: boolean) => void) | null = null;

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
      if (data.type === "task") {
        this.onTask?.(data as TaskMessage);
      } else if (data.type === "weights") {
        this.onWeights?.(data as WeightsMessage);
      } else if (data.type === "credited") {
        this.onCredited?.(data as CreditedMessage);
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

  sendReady() {
    this.ws?.readyState === WebSocket.OPEN &&
      this.ws.send(JSON.stringify({ type: "ready" }));
  }

  sendResult(taskId: string, cTileB64: string, computeTimeMs: number) {
    this.ws?.readyState === WebSocket.OPEN &&
      this.ws.send(
        JSON.stringify({
          type: "result",
          task_type: "tile",
          task_id: taskId,
          c_tile: cTileB64,
          compute_time_ms: computeTimeMs,
        })
      );
  }

  sendFFNResult(taskId: string, outputB64: string, computeTimeMs: number) {
    this.ws?.readyState === WebSocket.OPEN &&
      this.ws.send(
        JSON.stringify({
          type: "result",
          task_type: "ffn_forward",
          task_id: taskId,
          output: outputB64,
          compute_time_ms: computeTimeMs,
        })
      );
  }

  sendNeedWeights(layerIdx: number, weightsVersion: string) {
    this.ws?.readyState === WebSocket.OPEN &&
      this.ws.send(
        JSON.stringify({
          type: "need_weights",
          layer_idx: layerIdx,
          weights_version: weightsVersion,
        })
      );
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
