"""Shard replication manager — ensures weight redundancy across browser peers."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import N_LAYERS
from app.services.shard_registry import ShardRegistry

log = logging.getLogger(__name__)

MIN_REPLICAS = 2
CHECK_INTERVAL_S = 30.0


class ReplicationManager:
    def __init__(self, n_layers: int = N_LAYERS, min_replicas: int = MIN_REPLICAS):
        self.n_layers = n_layers
        self.min_replicas = min_replicas
        self._check_task: asyncio.Task | None = None

    def start(self, registry: ShardRegistry):
        if self._check_task is None:
            self._check_task = asyncio.create_task(self._check_loop(registry))

    def stop(self):
        if self._check_task:
            self._check_task.cancel()
            self._check_task = None

    async def handle_peer_disconnect(self, peer_id: str, registry: ShardRegistry, weights_version: str):
        """Reassign orphaned layers when a peer disconnects."""
        for layer_idx in range(self.n_layers):
            redundancy = registry.get_layer_redundancy(layer_idx, weights_version)
            if redundancy < self.min_replicas:
                await self.schedule_replication(layer_idx, weights_version, registry)

    async def on_peer_join(self, peer_id: str, manifest: list[dict], registry: ShardRegistry, weights_version: str):
        """Assign under-replicated layers to the new peer."""
        under_replicated: list[int] = []
        for layer_idx in range(self.n_layers):
            if not registry.peer_has_layer(peer_id, layer_idx, weights_version):
                redundancy = registry.get_layer_redundancy(layer_idx, weights_version)
                if redundancy < self.min_replicas:
                    under_replicated.append(layer_idx)

        for layer_idx in under_replicated:
            source = registry.get_best_shard_source(layer_idx, weights_version, exclude_peer=peer_id)
            ws = registry.get_peer_ws(peer_id)
            if ws:
                try:
                    await ws.send_json({
                        "type": "shard_transfer_request",
                        "layer_idx": layer_idx,
                        "source_peer_id": source.peer_id if source else None,
                        "version": weights_version,
                    })
                except Exception as e:
                    log.debug("Failed to request shard transfer to %s: %s", peer_id, e)

    async def schedule_replication(self, layer_idx: int, version: str, registry: ShardRegistry):
        """Pick a peer without the layer and tell it to download it."""
        all_peers = registry.get_all_peer_ids()
        peers_with = {p.peer_id for p in registry.get_peers_with_layer(layer_idx, version)}
        peers_without = [p for p in all_peers if p not in peers_with]

        if not peers_without:
            return

        target_peer = peers_without[0]
        source = registry.get_best_shard_source(layer_idx, version, exclude_peer=target_peer)
        ws = registry.get_peer_ws(target_peer)
        if ws:
            try:
                await ws.send_json({
                    "type": "shard_transfer_request",
                    "layer_idx": layer_idx,
                    "source_peer_id": source.peer_id if source else None,
                    "version": version,
                })
            except Exception as e:
                log.debug("Failed to schedule replication for layer %d: %s", layer_idx, e)

    async def check_redundancy(self, registry: ShardRegistry, weights_version: str):
        """Check all layers and trigger replication for under-replicated ones."""
        for layer_idx in range(self.n_layers):
            redundancy = registry.get_layer_redundancy(layer_idx, weights_version)
            if redundancy < self.min_replicas:
                await self.schedule_replication(layer_idx, weights_version, registry)

    async def _check_loop(self, registry: ShardRegistry):
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL_S)
                # Need weights version from the compute manager
                # This is called via the manager, so we just log for now
                log.debug("Replication check: %d peers registered", registry.peer_count)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug("Replication check error: %s", e)
