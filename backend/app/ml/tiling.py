import numpy as np

from app.core.config import TILE_SIZE


class TileTask:
    __slots__ = ("task_id", "a_tile", "b_tile", "i", "j", "k", "meta")

    def __init__(
        self,
        task_id: str,
        a_tile: np.ndarray,
        b_tile: np.ndarray,
        i: int,
        j: int,
        k: int,
        meta: dict,
    ):
        self.task_id = task_id
        self.a_tile = a_tile
        self.b_tile = b_tile
        self.i = i
        self.j = j
        self.k = k
        self.meta = meta


def decompose_matmul(
    a: np.ndarray,
    b: np.ndarray,
    step: int,
    layer: int,
    op_name: str,
    tile_size: int = TILE_SIZE,
) -> list[TileTask]:
    """Decompose C = A @ B into tile tasks.

    A: [M, K], B: [K, N] -> C: [M, N]
    """
    M, K = a.shape
    _, N = b.shape

    tasks = []
    n_i = (M + tile_size - 1) // tile_size
    n_j = (N + tile_size - 1) // tile_size
    n_k = (K + tile_size - 1) // tile_size

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                row_start = i * tile_size
                row_end = min(row_start + tile_size, M)
                col_start = j * tile_size
                col_end = min(col_start + tile_size, N)
                k_start = k * tile_size
                k_end = min(k_start + tile_size, K)

                a_tile = a[row_start:row_end, k_start:k_end]
                b_tile = b[k_start:k_end, col_start:col_end]

                # Pad to tile_size if needed
                if a_tile.shape != (tile_size, tile_size):
                    padded_a = np.zeros((tile_size, tile_size), dtype=np.float32)
                    padded_a[: a_tile.shape[0], : a_tile.shape[1]] = a_tile
                    a_tile = padded_a
                if b_tile.shape != (tile_size, tile_size):
                    padded_b = np.zeros((tile_size, tile_size), dtype=np.float32)
                    padded_b[: b_tile.shape[0], : b_tile.shape[1]] = b_tile
                    b_tile = padded_b

                task_id = f"s{step}_l{layer}_{op_name}_t{i}_{j}_{k}"
                tasks.append(
                    TileTask(
                        task_id=task_id,
                        a_tile=a_tile.astype(np.float32),
                        b_tile=b_tile.astype(np.float32),
                        i=i,
                        j=j,
                        k=k,
                        meta={"step": step, "layer": layer, "op": op_name},
                    )
                )
    return tasks


def assemble_tiles(
    tiles: dict[tuple[int, int, int], np.ndarray],
    M: int,
    N: int,
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """Assemble tile results back into full matrix C[M, N].

    tiles: dict of (i, j, k) -> C_tile[tile_size, tile_size]
    For same (i,j) with different k, sum the partial products.
    """
    result = np.zeros((M, N), dtype=np.float32)

    partials: dict[tuple[int, int], np.ndarray] = {}
    for (i, j, k), c_tile in tiles.items():
        key = (i, j)
        if key not in partials:
            partials[key] = np.zeros_like(c_tile)
        partials[key] += c_tile

    for (i, j), partial in partials.items():
        row_start = i * tile_size
        row_end = min(row_start + tile_size, M)
        col_start = j * tile_size
        col_end = min(col_start + tile_size, N)
        result[row_start:row_end, col_start:col_end] = partial[
            : row_end - row_start, : col_end - col_start
        ]

    return result
