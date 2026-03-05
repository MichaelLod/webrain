// Row-wise softmax: each workgroup handles one row
// Params: [n_cols, 0, 0, 0]

struct Params {
  n_cols: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  let n = params.n_cols;
  let offset = row * n;

  // Find max for numerical stability
  var max_val: f32 = -1e30;
  for (var i: u32 = 0u; i < n; i++) {
    max_val = max(max_val, data[offset + i]);
  }

  // Compute exp and sum
  var sum_exp: f32 = 0.0;
  for (var i: u32 = 0u; i < n; i++) {
    let e = exp(data[offset + i] - max_val);
    data[offset + i] = e;
    sum_exp += e;
  }

  // Normalize
  let inv_sum = 1.0 / sum_exp;
  for (var i: u32 = 0u; i < n; i++) {
    data[offset + i] = data[offset + i] * inv_sum;
  }
}
