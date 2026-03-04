// Tile matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// Workgroup processes a TILE_SIZE x TILE_SIZE output tile

struct Params {
  M: u32,
  K: u32,
  N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>; // 16x16
var<workgroup> tileB: array<f32, 256>; // 16x16

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.x;
  let col = gid.y;
  let localRow = lid.x;
  let localCol = lid.y;

  var sum: f32 = 0.0;
  let numTiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A
    let aRow = row;
    let aCol = t * TILE + localCol;
    if (aRow < params.M && aCol < params.K) {
      tileA[localRow * TILE + localCol] = a[aRow * params.K + aCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }

    // Load tile of B
    let bRow = t * TILE + localRow;
    let bCol = col;
    if (bRow < params.K && bCol < params.N) {
      tileB[localRow * TILE + localCol] = b[bRow * params.N + bCol];
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }

    workgroupBarrier();

    // Compute partial dot product
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      sum = sum + tileA[localRow * TILE + k] * tileB[k * TILE + localCol];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    c[row * params.N + col] = sum;
  }
}
