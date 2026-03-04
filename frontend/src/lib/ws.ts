const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export type TaskMessage = {
  type: "task";
  task_id: string;
  a_tile: string; // base64
  b_tile: string; // base64
  tile_size: number;
  position: { i: number; j: number; k: number };
  meta: { step: number; layer: number; op: string };
};

export type CreditedMessage = {
  type: "credited";
  tokens_earned: number;
};

export type ServerMessage = TaskMessage | CreditedMessage | { type: string };

export class ComputeWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  onTask: ((task: TaskMessage) => void) | null = null;
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
          task_id: taskId,
          c_tile: cTileB64,
          compute_time_ms: computeTimeMs,
        })
      );
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
