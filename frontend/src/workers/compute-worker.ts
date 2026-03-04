import { GPUEngine } from "./gpu-engine";

const engine = new GPUEngine();
let initialized = false;

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

  if (type === "destroy") {
    engine.destroy();
    initialized = false;
  }
};
