/**
 * Pipeline engine: runs full transformer layer ranges in the browser via WebGPU.
 * Handles attention + FFN (not just FFN slices like the swarm mode).
 */

import { GPUEngine } from "./gpu-engine";

export interface LayerWeights {
  // Attention
  qW: Float32Array; // [n_heads * head_dim, d_model]
  kW: Float32Array; // [n_kv_heads * head_dim, d_model]
  vW: Float32Array; // [n_kv_heads * head_dim, d_model]
  oW: Float32Array; // [d_model, n_heads * head_dim]
  ln1W: Float32Array; // [d_model]
  // FFN
  gateW: Float32Array; // [d_ff, d_model] transposed to [d_model, d_ff]
  upW: Float32Array;
  downW: Float32Array;
  ln2W: Float32Array; // [d_model]
}

export interface PipelineConfig {
  dModel: number;
  dFf: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  ropeTheta: number;
  rmsEps: number;
}

const DEFAULT_CONFIG: PipelineConfig = {
  dModel: 512,
  dFf: 1376,
  nHeads: 8,
  nKvHeads: 2,
  headDim: 64,
  ropeTheta: 10000.0,
  rmsEps: 1e-5,
};

/**
 * Computes RMSNorm on CPU (simpler than GPU dispatch for small tensors).
 */
function rmsNorm(
  x: Float32Array,
  weight: Float32Array,
  n: number,
  eps: number,
): Float32Array {
  const rows = x.length / n;
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    const offset = r * n;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      sumSq += x[offset + i] * x[offset + i];
    }
    const rms = 1.0 / Math.sqrt(sumSq / n + eps);
    for (let i = 0; i < n; i++) {
      out[offset + i] = x[offset + i] * rms * weight[i];
    }
  }
  return out;
}

/**
 * Applies SiLU element-wise: x * sigmoid(x)
 */
function silu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] / (1.0 + Math.exp(-x[i]));
  }
  return out;
}

/**
 * Element-wise multiply: a * b
 */
function elMul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i] * b[i];
  }
  return out;
}

/**
 * Residual add: a + b
 */
function residualAdd(
  a: Float32Array,
  b: Float32Array,
): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i] + b[i];
  }
  return out;
}

/**
 * Simple RoPE application on CPU for a sequence of positions.
 */
function applyRoPE(
  data: Float32Array,
  seqLen: number,
  headDim: number,
  startPos: number,
  theta: number,
): void {
  const halfDim = headDim / 2;
  for (let pos = 0; pos < seqLen; pos++) {
    for (let d = 0; d < halfDim; d++) {
      const freq = 1.0 / Math.pow(theta, (2 * d) / headDim);
      const angle = (pos + startPos) * freq;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const i0 = pos * headDim + d * 2;
      const i1 = i0 + 1;
      const x0 = data[i0];
      const x1 = data[i1];
      data[i0] = x0 * cosA - x1 * sinA;
      data[i1] = x0 * sinA + x1 * cosA;
    }
  }
}

/**
 * Row-wise softmax in place.
 */
function softmaxRows(data: Float32Array, rows: number, cols: number): void {
  for (let r = 0; r < rows; r++) {
    const offset = r * cols;
    let maxVal = -1e30;
    for (let c = 0; c < cols; c++) {
      if (data[offset + c] > maxVal) maxVal = data[offset + c];
    }
    let sumExp = 0;
    for (let c = 0; c < cols; c++) {
      data[offset + c] = Math.exp(data[offset + c] - maxVal);
      sumExp += data[offset + c];
    }
    const inv = 1.0 / sumExp;
    for (let c = 0; c < cols; c++) {
      data[offset + c] *= inv;
    }
  }
}

export class PipelineEngine {
  private engine: GPUEngine;
  private config: PipelineConfig;

  constructor(engine: GPUEngine, config?: Partial<PipelineConfig>) {
    this.engine = engine;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Run a range of transformer blocks on the input activations.
   * Returns the output activations after processing all layers.
   */
  async computeLayerRange(
    x: Float32Array,
    seqLen: number,
    layerWeights: LayerWeights[],
    startPos: number = 0,
  ): Promise<Float32Array> {
    const { dModel, dFf, nHeads, nKvHeads, headDim, ropeTheta, rmsEps } =
      this.config;
    let current = x;

    for (const lw of layerWeights) {
      current = await this.computeTransformerBlock(
        current,
        seqLen,
        lw,
        startPos,
        dModel,
        dFf,
        nHeads,
        nKvHeads,
        headDim,
        ropeTheta,
        rmsEps,
      );
    }

    return current;
  }

  private async computeTransformerBlock(
    x: Float32Array,
    seqLen: number,
    lw: LayerWeights,
    startPos: number,
    dModel: number,
    dFf: number,
    nHeads: number,
    nKvHeads: number,
    headDim: number,
    ropeTheta: number,
    rmsEps: number,
  ): Promise<Float32Array> {
    // --- Attention ---
    const normed1 = rmsNorm(x, lw.ln1W, dModel, rmsEps);

    // Q, K, V projections via GPU matmul
    const q = await this.engine.matmul(
      normed1,
      lw.qW,
      seqLen,
      dModel,
      nHeads * headDim,
    );
    const k = await this.engine.matmul(
      normed1,
      lw.kW,
      seqLen,
      dModel,
      nKvHeads * headDim,
    );
    const v = await this.engine.matmul(
      normed1,
      lw.vW,
      seqLen,
      dModel,
      nKvHeads * headDim,
    );

    // Apply RoPE to Q and K
    // For GQA, apply rope per head
    for (let h = 0; h < nHeads; h++) {
      const qHead = new Float32Array(
        q.buffer,
        0,
        q.length,
      );
      // Apply rope across the full Q (all heads concatenated, each head_dim-sized)
      const qSlice = q.subarray(0, seqLen * nHeads * headDim);
      // RoPE is applied per-head, we simplify by applying to chunks
      for (let s = 0; s < seqLen; s++) {
        const qOff = s * nHeads * headDim + h * headDim;
        const halfDim = headDim / 2;
        for (let d = 0; d < halfDim; d++) {
          const freq = 1.0 / Math.pow(ropeTheta, (2 * d) / headDim);
          const angle = (s + startPos) * freq;
          const cosA = Math.cos(angle);
          const sinA = Math.sin(angle);
          const i0 = qOff + d * 2;
          const i1 = i0 + 1;
          const x0 = q[i0], x1 = q[i1];
          q[i0] = x0 * cosA - x1 * sinA;
          q[i1] = x0 * sinA + x1 * cosA;
        }
      }
    }

    for (let h = 0; h < nKvHeads; h++) {
      for (let s = 0; s < seqLen; s++) {
        const kOff = s * nKvHeads * headDim + h * headDim;
        const halfDim = headDim / 2;
        for (let d = 0; d < halfDim; d++) {
          const freq = 1.0 / Math.pow(ropeTheta, (2 * d) / headDim);
          const angle = (s + startPos) * freq;
          const cosA = Math.cos(angle);
          const sinA = Math.sin(angle);
          const i0 = kOff + d * 2;
          const i1 = i0 + 1;
          const x0 = k[i0], x1 = k[i1];
          k[i0] = x0 * cosA - x1 * sinA;
          k[i1] = x0 * sinA + x1 * cosA;
        }
      }
    }

    // Scaled dot-product attention with GQA
    const headsPerGroup = nHeads / nKvHeads;
    const attnOut = new Float32Array(seqLen * nHeads * headDim);
    const scale = 1.0 / Math.sqrt(headDim);

    for (let h = 0; h < nHeads; h++) {
      const kvHead = Math.floor(h / headsPerGroup);

      // Compute attention scores for this head
      const scores = new Float32Array(seqLen * seqLen);
      for (let sq = 0; sq < seqLen; sq++) {
        for (let sk = 0; sk < seqLen; sk++) {
          // Causal mask
          if (sk > sq + startPos) {
            scores[sq * seqLen + sk] = -1e9;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot +=
              q[sq * nHeads * headDim + h * headDim + d] *
              k[sk * nKvHeads * headDim + kvHead * headDim + d];
          }
          scores[sq * seqLen + sk] = dot * scale;
        }
      }

      // Softmax
      softmaxRows(scores, seqLen, seqLen);

      // Weighted sum of values
      for (let sq = 0; sq < seqLen; sq++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let sk = 0; sk < seqLen; sk++) {
            sum +=
              scores[sq * seqLen + sk] *
              v[sk * nKvHeads * headDim + kvHead * headDim + d];
          }
          attnOut[sq * nHeads * headDim + h * headDim + d] = sum;
        }
      }
    }

    // Output projection
    const attnProjected = await this.engine.matmul(
      attnOut,
      lw.oW,
      seqLen,
      nHeads * headDim,
      dModel,
    );

    // Residual add
    const afterAttn = residualAdd(x, attnProjected);

    // --- FFN ---
    const normed2 = rmsNorm(afterAttn, lw.ln2W, dModel, rmsEps);

    const ffnOut = await this.engine.computeFFN(
      normed2,
      lw.gateW,
      lw.upW,
      lw.downW,
      seqLen,
      dModel,
      dFf,
    );

    // Residual add
    return residualAdd(afterAttn, ffnOut);
  }
}
