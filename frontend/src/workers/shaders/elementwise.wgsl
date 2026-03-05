// Element-wise operations: multiply, add, silu
// op: 0 = multiply (a*b), 1 = add (a+b), 2 = silu (a * sigmoid(a)), 3 = residual add (a+b)

struct Params {
  n: u32,
  op: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) {
    return;
  }

  if (params.op == 0u) {
    // element-wise multiply
    a[idx] = a[idx] * b[idx];
  } else if (params.op == 1u) {
    // element-wise add
    a[idx] = a[idx] + b[idx];
  } else if (params.op == 2u) {
    // SiLU: x * sigmoid(x)
    let x = a[idx];
    a[idx] = x / (1.0 + exp(-x));
  } else if (params.op == 3u) {
    // residual add (same as add, separate for clarity)
    a[idx] = a[idx] + b[idx];
  }
}
