"""WebRTC signaling over the existing WebSocket connection.

Routes offer/answer/ICE candidates between browser peers without
requiring any new server endpoints.
"""

from __future__ import annotations

import logging

from fastapi import WebSocket

from app.services.shard_registry import ShardRegistry

log = logging.getLogger(__name__)


class SignalingService:
    async def handle_signal(self, from_peer: str, data: dict, registry: ShardRegistry):
        signal_type = data.get("type", "")
        target_peer = data.get("target_peer", "")

        if not target_peer:
            return

        target_ws = registry.get_peer_ws(target_peer)
        if not target_ws:
            log.debug("Signaling target %s not found", target_peer)
            return

        try:
            await target_ws.send_json({
                "type": signal_type,
                "from_peer": from_peer,
                **{k: v for k, v in data.items() if k not in ("type", "target_peer")},
            })
        except Exception as e:
            log.debug("Failed to relay signal to %s: %s", target_peer, e)

    async def initiate_connection(self, peer_a: str, peer_b: str, registry: ShardRegistry):
        """Tell peer_a to create an offer for peer_b."""
        ws_a = registry.get_peer_ws(peer_a)
        if not ws_a:
            return

        try:
            await ws_a.send_json({
                "type": "webrtc_connect",
                "target_peer": peer_b,
            })
        except Exception as e:
            log.debug("Failed to initiate WebRTC %s -> %s: %s", peer_a, peer_b, e)
