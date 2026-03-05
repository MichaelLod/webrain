import { GPUEngine } from "./gpu-engine";
import { WeightStore, type ShardManifest } from "../lib/weight-store";
import { PipelineEngine, type LayerWeights, type PipelineConfig } from "./pipeline-engine";

const engine = new GPUEngine();
const weightStore = new WeightStore();
let pipelineEngine: PipelineEngine | null = null;
let initialized = false;

// In-memory weight cache: key = `${weightsVersion}_${layerIdx}` -> { gateT, upT, downT }
const weightCache = new Map<
  string,
  { gateT: Float32Array; upT: Float32Array; downT: Float32Array }
>();

// Full layer weights for pipeline mode
const pipelineWeights = new Map<number, LayerWeights>();

function cacheKey(version: string, layer: number): string {
  return `${version}_${layer}`;
}

async function initWeightStore() {
  try {
    await weightStore.open();
  } catch {
    // IndexedDB not available (e.g., in some worker contexts)
  }
}

async function persistWeights(
  layerIdx: number,
  version: string,
  gate: ArrayBuffer,
  up: ArrayBuffer,
  down: ArrayBuffer,
) {
  try {
    await Promise.all([
      weightStore.storeShard(layerIdx, "gate", version, gate),
      weightStore.storeShard(layerIdx, "up", version, up),
      weightStore.storeShard(layerIdx, "down", version, down),
    ]);
  } catch {
    // Silently fail — in-memory cache is the primary store
  }
}

async function loadFromIndexedDB(
  layerIdx: number,
  version: string,
): Promise<{ gateT: Float32Array; upT: Float32Array; downT: Float32Array } | null> {
  try {
    const [gate, up, down] = await Promise.all([
      weightStore.getShard(layerIdx, "gate", version),
      weightStore.getShard(layerIdx, "up", version),
      weightStore.getShard(layerIdx, "down", version),
    ]);
    if (gate && up && down) {
      return {
        gateT: new Float32Array(gate),
        upT: new Float32Array(up),
        downT: new Float32Array(down),
      };
    }
  } catch {
    // IndexedDB not available
  }
  return null;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === "init") {
    await initWeightStore();
    const result = await engine.init();
    initialized = result.supported;
    if (initialized) {
      pipelineEngine = new PipelineEngine(engine);
    }
    // Load manifest from IndexedDB and send back
    const manifest = await weightStore.getManifest();
    self.postMessage({ type: "init_result", payload: { ...result, manifest } });
    return;
  }

  if (type === "get_manifest") {
    const manifest = await weightStore.getManifest();
    self.postMessage({ type: "manifest", payload: { manifest } });
    return;
  }

  if (type === "compute" && initialized) {
    const { taskId, aTile, bTile, tileSize } = payload;

    const a = new Float32Array(aTile);
    const b = new Float32Array(bTile);

    const start = performance.now();
    try {
      const result = await engine.matmul(a, b, tileSize, tileSize, tileSize);
      const computeTimeMs = performance.now() - start;

      self.postMessage({
        type: "result",
        payload: {
          taskId,
          taskType: "tile",
          cTile: result.buffer,
          computeTimeMs,
        },
      });
    } catch (err) {
      self.postMessage({
        type: "error",
        payload: { taskId, error: String(err) },
      });
    }
    return;
  }

  if (type === "ffn_forward" && initialized) {
    const {
      taskId,
      activations,
      layerIdx,
      dModel,
      dFf,
      seqLen,
      weightsVersion,
      weights,
    } = payload;

    const key = cacheKey(weightsVersion, layerIdx);

    // Cache weights if provided (in memory + IndexedDB)
    if (weights) {
      const gateT = new Float32Array(weights.gate);
      const upT = new Float32Array(weights.up);
      const downT = new Float32Array(weights.down);
      weightCache.set(key, { gateT, upT, downT });
      persistWeights(layerIdx, weightsVersion, weights.gate, weights.up, weights.down);
    }

    let cached = weightCache.get(key);

    // Try IndexedDB if not in memory
    if (!cached) {
      const fromDb = await loadFromIndexedDB(layerIdx, weightsVersion);
      if (fromDb) {
        weightCache.set(key, fromDb);
        cached = fromDb;
      }
    }

    if (!cached) {
      self.postMessage({
        type: "need_weights",
        payload: { layerIdx, weightsVersion, taskId },
      });
      return;
    }

    const x = new Float32Array(activations);
    const M = seqLen;

    const start = performance.now();
    try {
      const result = await engine.computeFFN(
        x,
        cached.gateT,
        cached.upT,
        cached.downT,
        M,
        dModel,
        dFf,
      );
      const computeTimeMs = performance.now() - start;

      self.postMessage({
        type: "result",
        payload: {
          taskId,
          taskType: "ffn_forward",
          output: result.buffer,
          computeTimeMs,
        },
      });
    } catch (err) {
      self.postMessage({
        type: "error",
        payload: { taskId, error: String(err) },
      });
    }
    return;
  }

  if (type === "cache_weights") {
    const { layerIdx, weightsVersion, gate, up, down } = payload;
    const key = cacheKey(weightsVersion, layerIdx);
    const gateT = new Float32Array(gate);
    const upT = new Float32Array(up);
    const downT = new Float32Array(down);
    weightCache.set(key, { gateT, upT, downT });
    persistWeights(layerIdx, weightsVersion, gate, up, down);
    self.postMessage({ type: "weights_cached", payload: { layerIdx, weightsVersion } });
    return;
  }

  if (type === "pipeline_forward" && initialized && pipelineEngine) {
    const { requestId, activations, startLayer, endLayer, seqLen, dModel, startPos } = payload;

    // Collect layer weights for the assigned range
    const layers: LayerWeights[] = [];
    for (let l = startLayer; l < endLayer; l++) {
      const lw = pipelineWeights.get(l);
      if (!lw) {
        // Missing weights — report error
        self.postMessage({
          type: "pipeline_error",
          payload: { requestId, error: `Missing weights for layer ${l}` },
        });
        return;
      }
      layers.push(lw);
    }

    const x = new Float32Array(activations);
    const start = performance.now();
    try {
      const result = await pipelineEngine.computeLayerRange(x, seqLen, layers, startPos);
      const computeTimeMs = performance.now() - start;

      self.postMessage({
        type: "pipeline_result",
        payload: {
          requestId,
          activations: result.buffer,
          computeTimeMs,
        },
      });
    } catch (err) {
      self.postMessage({
        type: "pipeline_error",
        payload: { requestId, error: String(err) },
      });
    }
    return;
  }

  if (type === "training_forward" && initialized && pipelineEngine) {
    const { requestId, activations, startLayer, endLayer, seqLen, dModel, dFf } = payload;

    const layers: LayerWeights[] = [];
    for (let l = startLayer; l < endLayer; l++) {
      const lw = pipelineWeights.get(l);
      if (!lw) {
        self.postMessage({
          type: "training_error",
          payload: { requestId, error: `Missing weights for layer ${l}` },
        });
        return;
      }
      layers.push(lw);
    }

    const x = new Float32Array(activations);
    const start = performance.now();
    try {
      const result = await pipelineEngine.computeLayerRange(x, seqLen, layers, 0);
      const computeTimeMs = performance.now() - start;

      self.postMessage({
        type: "training_forward_result",
        payload: {
          requestId,
          activations: result.buffer,
          computeTimeMs,
        },
      });
    } catch (err) {
      self.postMessage({
        type: "training_error",
        payload: { requestId, error: String(err) },
      });
    }
    return;
  }

  if (type === "store_pipeline_weights") {
    // Store full layer weights for pipeline mode
    const { layerIdx, weights } = payload;
    pipelineWeights.set(layerIdx, {
      qW: new Float32Array(weights.qW),
      kW: new Float32Array(weights.kW),
      vW: new Float32Array(weights.vW),
      oW: new Float32Array(weights.oW),
      ln1W: new Float32Array(weights.ln1W),
      gateW: new Float32Array(weights.gateW),
      upW: new Float32Array(weights.upW),
      downW: new Float32Array(weights.downW),
      ln2W: new Float32Array(weights.ln2W),
    });
    self.postMessage({ type: "pipeline_weights_stored", payload: { layerIdx } });
    return;
  }

  if (type === "destroy") {
    engine.destroy();
    initialized = false;
    pipelineEngine = null;
    weightCache.clear();
    pipelineWeights.clear();
  }
};
