// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
// Params uniform: [N, eps, 0, 0]

struct Params {
  n: u32,
  eps_bits: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  let n = params.n;
  let eps = bitcast<f32>(params.eps_bits);

  let offset = row * n;

  // Compute mean of squares
  var sum_sq: f32 = 0.0;
  for (var i: u32 = 0u; i < n; i++) {
    let v = input[offset + i];
    sum_sq += v * v;
  }
  let rms = 1.0 / sqrt(sum_sq / f32(n) + eps);

  // Normalize and scale
  for (var i: u32 = 0u; i < n; i++) {
    output[offset + i] = input[offset + i] * rms * weight[i];
  }
}
