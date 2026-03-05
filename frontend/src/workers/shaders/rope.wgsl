// Rotary Position Embedding (RoPE)
// Applies rotation to pairs of dimensions: (x0,x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
// Params: [seq_len, head_dim, start_pos, theta_bits]

struct Params {
  seq_len: u32,
  head_dim: u32,
  start_pos: u32,
  theta_bits: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.seq_len * half_dim;

  if (idx >= total_pairs) {
    return;
  }

  let pos_in_seq = idx / half_dim;
  let dim_pair = idx % half_dim;

  let theta = bitcast<f32>(params.theta_bits);
  let position = f32(pos_in_seq + params.start_pos);
  let freq = 1.0 / pow(theta, f32(2u * dim_pair) / f32(params.head_dim));
  let angle = position * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base_offset = pos_in_seq * params.head_dim;
  let i0 = base_offset + dim_pair * 2u;
  let i1 = i0 + 1u;

  let x0 = data[i0];
  let x1 = data[i1];
  data[i0] = x0 * cos_a - x1 * sin_a;
  data[i1] = x0 * sin_a + x1 * cos_a;
}
