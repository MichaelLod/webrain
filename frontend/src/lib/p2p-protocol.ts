/**
 * Binary protocol for efficient DataChannel transfer between browser peers.
 *
 * Format: [4B type][4B id_len][id bytes][4B payload_len][payload bytes]
 * Message types: 0x01 activations, 0x02 kv_cache, 0x03 gradients, 0x04 weight_shard
 */

export const MSG_ACTIVATIONS = 0x01;
export const MSG_KV_CACHE = 0x02;
export const MSG_GRADIENTS = 0x03;
export const MSG_WEIGHT_SHARD = 0x04;

export interface P2PMessage {
  type: number;
  id: string;
  payload: ArrayBuffer;
}

export function encodeActivationMessage(
  requestId: string,
  payload: ArrayBuffer,
): ArrayBuffer {
  return encodeMessage(MSG_ACTIVATIONS, requestId, payload);
}

export function encodeMessage(
  type: number,
  id: string,
  payload: ArrayBuffer,
): ArrayBuffer {
  const encoder = new TextEncoder();
  const idBytes = encoder.encode(id);

  const totalLen = 4 + 4 + idBytes.byteLength + 4 + payload.byteLength;
  const buffer = new ArrayBuffer(totalLen);
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  let offset = 0;

  // Type (4 bytes, big-endian)
  view.setUint32(offset, type);
  offset += 4;

  // ID length + ID bytes
  view.setUint32(offset, idBytes.byteLength);
  offset += 4;
  u8.set(idBytes, offset);
  offset += idBytes.byteLength;

  // Payload length + payload bytes
  view.setUint32(offset, payload.byteLength);
  offset += 4;
  u8.set(new Uint8Array(payload), offset);

  return buffer;
}

export function decodeMessage(buffer: ArrayBuffer): P2PMessage {
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);
  const decoder = new TextDecoder();

  let offset = 0;

  const type = view.getUint32(offset);
  offset += 4;

  const idLen = view.getUint32(offset);
  offset += 4;
  const id = decoder.decode(u8.slice(offset, offset + idLen));
  offset += idLen;

  const payloadLen = view.getUint32(offset);
  offset += 4;
  const payload = buffer.slice(offset, offset + payloadLen);

  return { type, id, payload };
}
