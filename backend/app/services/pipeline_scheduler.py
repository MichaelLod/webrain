"""Pipeline stage assignment for distributed layer-parallel inference."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.core.config import N_LAYERS
from app.services.shard_registry import ShardRegistry

log = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    stage_id: int
    layer_range: tuple[int, int]  # (start_inclusive, end_exclusive)
    assigned_peers: list[str] = field(default_factory=list)
    primary_peer: str | None = None
    backup_peers: list[str] = field(default_factory=list)


class PipelineScheduler:
    def __init__(self, n_layers: int = N_LAYERS):
        self.n_layers = n_layers
        self.stages: list[PipelineStage] = []
        self._peer_assignments: dict[str, tuple[int, int]] = {}

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def is_active(self) -> bool:
        return len(self.stages) > 0 and all(s.primary_peer for s in self.stages)

    def recompute_assignment(self, registry: ShardRegistry) -> list[PipelineStage]:
        peer_ids = registry.get_all_peer_ids()
        n_peers = len(peer_ids)

        if n_peers < 2:
            self.stages = []
            self._peer_assignments.clear()
            return self.stages

        n_stages = min(n_peers, math.ceil(self.n_layers / 3))
        layers_per_stage = math.ceil(self.n_layers / n_stages)

        stages: list[PipelineStage] = []
        for s in range(n_stages):
            start = s * layers_per_stage
            end = min(start + layers_per_stage, self.n_layers)
            stages.append(PipelineStage(stage_id=s, layer_range=(start, end)))

        # Score peers for each stage: prefer those already caching relevant layers
        for stage in stages:
            start, end = stage.layer_range
            scored: list[tuple[str, int]] = []
            for pid in peer_ids:
                score = sum(
                    1 for l in range(start, end)
                    if registry.get_layer_redundancy(l) > 0
                    and any(p.peer_id == pid for p in registry.get_peers_with_layer(l))
                )
                scored.append((pid, score))
            scored.sort(key=lambda x: -x[1])
            stage.assigned_peers = [p for p, _ in scored]

        # Assign primary peers round-robin, avoiding double-assignment
        assigned_primary: set[str] = set()
        for stage in stages:
            for pid in stage.assigned_peers:
                if pid not in assigned_primary:
                    stage.primary_peer = pid
                    assigned_primary.add(pid)
                    break
            stage.backup_peers = [
                p for p in stage.assigned_peers if p != stage.primary_peer
            ]

        # If not enough unique primaries, allow sharing
        for stage in stages:
            if stage.primary_peer is None and stage.assigned_peers:
                stage.primary_peer = stage.assigned_peers[0]

        self.stages = stages
        self._peer_assignments.clear()
        for stage in stages:
            if stage.primary_peer:
                self._peer_assignments[stage.primary_peer] = stage.layer_range
            for bp in stage.backup_peers:
                if bp not in self._peer_assignments:
                    self._peer_assignments[bp] = stage.layer_range

        log.info(
            "Pipeline: %d stages across %d peers — %s",
            n_stages, n_peers,
            [(s.stage_id, s.layer_range, s.primary_peer) for s in stages],
        )
        return stages

    def get_peer_assignment(self, peer_id: str) -> tuple[int, int] | None:
        return self._peer_assignments.get(peer_id)

    def get_stage_for_layer(self, layer_idx: int) -> PipelineStage | None:
        for stage in self.stages:
            start, end = stage.layer_range
            if start <= layer_idx < end:
                return stage
        return None

    def get_next_stage(self, stage_id: int) -> PipelineStage | None:
        idx = stage_id + 1
        if idx < len(self.stages):
            return self.stages[idx]
        return None
