import { GPUEngine } from "./gpu-engine";

const engine = new GPUEngine();
let initialized = false;

// Weight cache: key = `${weightsVersion}_${layerIdx}` -> { gateT, upT, downT }
const weightCache = new Map<
  string,
  { gateT: Float32Array; upT: Float32Array; downT: Float32Array }
>();

function cacheKey(version: string, layer: number): string {
  return `${version}_${layer}`;
}

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === "init") {
    const result = await engine.init();
    initialized = result.supported;
    self.postMessage({ type: "init_result", payload: result });
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
      weights, // may be undefined if cached
    } = payload;

    const key = cacheKey(weightsVersion, layerIdx);

    // Cache weights if provided
    if (weights) {
      weightCache.set(key, {
        gateT: new Float32Array(weights.gate),
        upT: new Float32Array(weights.up),
        downT: new Float32Array(weights.down),
      });
    }

    const cached = weightCache.get(key);
    if (!cached) {
      // Request weights from server
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
    // Direct weight cache update (from server weight response)
    const { layerIdx, weightsVersion, gate, up, down } = payload;
    const key = cacheKey(weightsVersion, layerIdx);
    weightCache.set(key, {
      gateT: new Float32Array(gate),
      upT: new Float32Array(up),
      downT: new Float32Array(down),
    });
    self.postMessage({ type: "weights_cached", payload: { layerIdx, weightsVersion } });
    return;
  }

  if (type === "destroy") {
    engine.destroy();
    initialized = false;
    weightCache.clear();
  }
};
