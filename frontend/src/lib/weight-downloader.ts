/**
 * Progressive weight download with fp16->fp32 conversion.
 * Downloads shard files via HTTP (more efficient than WebSocket for large transfers).
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ShardManifestEntry {
  name: string;
  type: "embedding" | "layers" | "head";
  layers: [number, number] | null;
  size_bytes: number;
  dtype: string;
}

export interface ShardManifest {
  format: string;
  layers_per_shard: number;
  n_layers: number;
  shards: ShardManifestEntry[];
}

/**
 * Convert a Float16 ArrayBuffer to Float32Array.
 * Handles the fp16->fp32 expansion needed for WebGPU.
 */
export function fp16ToFp32(data: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(data);
  const f32 = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const bits = u16[i];
    const sign = (bits >> 15) & 1;
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;

    let value: number;
    if (exponent === 0) {
      // Subnormal or zero
      value = (mantissa / 1024) * Math.pow(2, -14);
    } else if (exponent === 31) {
      // Inf or NaN
      value = mantissa === 0 ? Infinity : NaN;
    } else {
      value = (1 + mantissa / 1024) * Math.pow(2, exponent - 15);
    }
    f32[i] = sign ? -value : value;
  }
  return f32;
}

export async function fetchShardManifest(): Promise<ShardManifest | null> {
  try {
    const res = await fetch(`${API_URL}/api/v1/weights/shards/manifest`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/**
 * Download a specific shard by name with progress callback.
 */
export async function downloadShard(
  shardName: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(`${API_URL}/api/v1/weights/shards/${shardName}`);
  if (!res.ok) throw new Error(`Failed to download shard: ${shardName}`);

  const total = parseInt(res.headers.get("content-length") || "0", 10);
  const reader = res.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }

  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result.buffer;
}

/**
 * Download layers for a specific range.
 */
export async function downloadLayerRange(
  startLayer: number,
  endLayer: number,
  manifest: ShardManifest,
  onProgress?: (loaded: number, total: number, shardName: string) => void,
): Promise<Map<string, ArrayBuffer>> {
  const results = new Map<string, ArrayBuffer>();
  const relevantShards = manifest.shards.filter((s) => {
    if (s.type !== "layers" || !s.layers) return false;
    const [sStart, sEnd] = s.layers;
    return sStart < endLayer && sEnd > startLayer;
  });

  for (const shard of relevantShards) {
    const data = await downloadShard(shard.name, (loaded, total) => {
      onProgress?.(loaded, total, shard.name);
    });
    results.set(shard.name, data);
  }
  return results;
}

/**
 * Request a shard transfer from another browser peer via WebRTC DataChannel.
 */
export async function downloadFromPeer(
  peerId: string,
  layerIdx: number,
  sendRequest: (peerId: string, layerIdx: number) => void,
): Promise<void> {
  sendRequest(peerId, layerIdx);
  // The actual data arrives via the WebRTC message handler
}
