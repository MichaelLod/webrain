"""Registry tracking which browser peer has which weight shards cached."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastapi import WebSocket

log = logging.getLogger(__name__)


@dataclass
class ShardEntry:
    version: str
    components: list[str]
    total_bytes: int = 0


@dataclass
class PeerShardInfo:
    peer_id: str
    ws: WebSocket
    user_id: int
    layers: dict[int, ShardEntry] = field(default_factory=dict)


class ShardRegistry:
    def __init__(self):
        self._peers: dict[str, PeerShardInfo] = {}
        self._ws_to_peer: dict[WebSocket, str] = {}

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    def register_peer(self, peer_id: str, ws: WebSocket, user_id: int) -> PeerShardInfo:
        info = PeerShardInfo(peer_id=peer_id, ws=ws, user_id=user_id)
        self._peers[peer_id] = info
        self._ws_to_peer[ws] = peer_id
        return info

    def unregister_peer(self, peer_id: str):
        info = self._peers.pop(peer_id, None)
        if info:
            self._ws_to_peer.pop(info.ws, None)

    def unregister_by_ws(self, ws: WebSocket) -> str | None:
        peer_id = self._ws_to_peer.pop(ws, None)
        if peer_id:
            self._peers.pop(peer_id, None)
        return peer_id

    def get_peer_id(self, ws: WebSocket) -> str | None:
        return self._ws_to_peer.get(ws)

    def get_peer(self, peer_id: str) -> PeerShardInfo | None:
        return self._peers.get(peer_id)

    def update_peer_shards(self, peer_id: str, manifest: list[dict]):
        info = self._peers.get(peer_id)
        if not info:
            return
        info.layers.clear()
        for entry in manifest:
            layer_idx = entry.get("layerIdx", entry.get("layer_idx", -1))
            if layer_idx < 0:
                continue
            info.layers[layer_idx] = ShardEntry(
                version=entry.get("version", ""),
                components=entry.get("components", []),
                total_bytes=entry.get("totalBytes", entry.get("total_bytes", 0)),
            )

    def get_peers_with_layer(self, layer_idx: int, version: str | None = None) -> list[PeerShardInfo]:
        result = []
        for info in self._peers.values():
            entry = info.layers.get(layer_idx)
            if entry and (version is None or entry.version == version):
                result.append(info)
        return result

    def get_layer_redundancy(self, layer_idx: int, version: str | None = None) -> int:
        return len(self.get_peers_with_layer(layer_idx, version))

    def get_best_shard_source(self, layer_idx: int, version: str, exclude_peer: str | None = None) -> PeerShardInfo | None:
        candidates = self.get_peers_with_layer(layer_idx, version)
        for c in candidates:
            if c.peer_id != exclude_peer:
                return c
        return None

    def peer_has_layer(self, peer_id: str, layer_idx: int, version: str) -> bool:
        info = self._peers.get(peer_id)
        if not info:
            return False
        entry = info.layers.get(layer_idx)
        return entry is not None and entry.version == version

    def get_all_peer_ids(self) -> list[str]:
        return list(self._peers.keys())

    def get_peer_ws(self, peer_id: str) -> WebSocket | None:
        info = self._peers.get(peer_id)
        return info.ws if info else None
