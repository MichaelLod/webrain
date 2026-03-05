/**
 * WebRTC data channel manager for peer-to-peer activation transfer.
 * Falls back to WebSocket relay if WebRTC connection fails.
 */

import {
  encodeActivationMessage,
  decodeMessage,
  MSG_ACTIVATIONS,
  type P2PMessage,
} from "./p2p-protocol";

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

const CHANNEL_NAME = "activations";

interface PeerConnection {
  pc: RTCPeerConnection;
  dc: RTCDataChannel | null;
  peerId: string;
  ready: boolean;
}

type MessageHandler = (peerId: string, msg: P2PMessage) => void;
type SignalSender = (targetPeerId: string, data: unknown) => void;

export class WebRTCManager {
  private connections = new Map<string, PeerConnection>();
  private onMessage: MessageHandler | null = null;
  private sendSignal: SignalSender;

  constructor(sendSignal: SignalSender) {
    this.sendSignal = sendSignal;
  }

  setMessageHandler(handler: MessageHandler) {
    this.onMessage = handler;
  }

  async createOffer(targetPeerId: string): Promise<void> {
    const conn = this.getOrCreateConnection(targetPeerId);

    // Create data channel (offerer creates it)
    const dc = conn.pc.createDataChannel(CHANNEL_NAME, {
      ordered: true,
    });
    this.setupDataChannel(dc, targetPeerId, conn);
    conn.dc = dc;

    const offer = await conn.pc.createOffer();
    await conn.pc.setLocalDescription(offer);

    this.sendSignal(targetPeerId, {
      type: "webrtc_offer",
      sdp: offer.sdp,
    });
  }

  async handleOffer(
    fromPeerId: string,
    sdp: string,
  ): Promise<void> {
    const conn = this.getOrCreateConnection(fromPeerId);

    // Answerer receives data channel via event
    conn.pc.ondatachannel = (event) => {
      const dc = event.channel;
      this.setupDataChannel(dc, fromPeerId, conn);
      conn.dc = dc;
    };

    await conn.pc.setRemoteDescription(
      new RTCSessionDescription({ type: "offer", sdp }),
    );
    const answer = await conn.pc.createAnswer();
    await conn.pc.setLocalDescription(answer);

    this.sendSignal(fromPeerId, {
      type: "webrtc_answer",
      sdp: answer.sdp,
    });
  }

  async handleAnswer(fromPeerId: string, sdp: string): Promise<void> {
    const conn = this.connections.get(fromPeerId);
    if (!conn) return;
    await conn.pc.setRemoteDescription(
      new RTCSessionDescription({ type: "answer", sdp }),
    );
  }

  handleIceCandidate(
    fromPeerId: string,
    candidate: RTCIceCandidateInit,
  ): void {
    const conn = this.connections.get(fromPeerId);
    if (!conn) return;
    conn.pc.addIceCandidate(new RTCIceCandidate(candidate));
  }

  sendActivations(
    targetPeerId: string,
    requestId: string,
    activations: ArrayBuffer,
  ): boolean {
    const conn = this.connections.get(targetPeerId);
    if (!conn?.dc || !conn.ready) return false;

    try {
      const msg = encodeActivationMessage(requestId, activations);
      conn.dc.send(msg);
      return true;
    } catch {
      return false;
    }
  }

  isConnected(peerId: string): boolean {
    const conn = this.connections.get(peerId);
    return conn?.ready ?? false;
  }

  getConnectedPeerIds(): string[] {
    const ids: string[] = [];
    for (const [id, conn] of this.connections) {
      if (conn.ready) ids.push(id);
    }
    return ids;
  }

  disconnect(peerId: string): void {
    const conn = this.connections.get(peerId);
    if (conn) {
      conn.dc?.close();
      conn.pc.close();
      this.connections.delete(peerId);
    }
  }

  disconnectAll(): void {
    for (const [id] of this.connections) {
      this.disconnect(id);
    }
  }

  private getOrCreateConnection(peerId: string): PeerConnection {
    let conn = this.connections.get(peerId);
    if (conn) return conn;

    const pc = new RTCPeerConnection(RTC_CONFIG);
    conn = { pc, dc: null, peerId, ready: false };
    this.connections.set(peerId, conn);

    // ICE candidate forwarding
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendSignal(peerId, {
          type: "webrtc_ice",
          candidate: event.candidate.toJSON(),
        });
      }
    };

    pc.onconnectionstatechange = () => {
      if (
        pc.connectionState === "failed" ||
        pc.connectionState === "disconnected"
      ) {
        this.disconnect(peerId);
      }
    };

    return conn;
  }

  private setupDataChannel(
    dc: RTCDataChannel,
    peerId: string,
    conn: PeerConnection,
  ): void {
    dc.binaryType = "arraybuffer";

    dc.onopen = () => {
      conn.ready = true;
    };

    dc.onclose = () => {
      conn.ready = false;
    };

    dc.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const msg = decodeMessage(event.data);
        this.onMessage?.(peerId, msg);
      }
    };
  }
}
