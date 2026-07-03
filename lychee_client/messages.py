"""Message model: parsing lifecycle messages and state updates."""

from __future__ import annotations

from typing import Any


class StartMessage:
    """Parsed 'start' message from the server."""

    def __init__(self, data: dict[str, Any]):
        self.raw = data
        self.match_id: str = data.get("matchId", "")
        self.duration_round: int = data.get("durationRound", 600)
        self.round: int = data.get("round", 1)
        self.players: list[dict] = data.get("players", [])
        self.nodes: list[dict] = data.get("nodes", [])
        self.edges: list[dict] = data.get("edges", [])
        self.resources: list[dict] = data.get("resources", [])
        self.task_templates: list[dict] = data.get("taskTemplates", [])
        map_data = data.get("map", {})
        gameplay = map_data.get("gameplay", {})
        roles = gameplay.get("roles", {})
        self.start_node_id: str = roles.get("startNodeId", "")
        self.gate_node_id: str = roles.get("gateNodeId", "")
        self.terminal_node_ids: list[str] = roles.get("terminalNodeIds", [])

    def find_self_player(self, player_id: int) -> dict | None:
        for p in self.players:
            if p.get("playerId") == player_id:
                return p
        return None


class InquireMessage:
    """Parsed 'inquire' message from the server."""

    def __init__(self, data: dict[str, Any]):
        self.raw = data
        self.match_id: str = data.get("matchId", "")
        self.round: int = data.get("round", 0)
        self.phase: str = data.get("phase", "")
        self.players: list[dict] = data.get("players", [])
        self.nodes: list[dict] = data.get("nodes", [])
        self.edges: list[dict] = data.get("edges", [])
        self.weather: dict = data.get("weather", {})
        self.tasks: list[dict] = data.get("tasks", [])
        self.bounties: list[dict] = data.get("bounties", [])
        self.contests: list[dict] = data.get("contests", [])
        self.events: list[dict] = data.get("events", [])
        self.action_results: list[dict] = data.get("actionResults", [])
        self.score_preview: dict = data.get("scorePreview", {})
        self.debug: dict = data.get("debug", {})

    def find_self_player(self, player_id: int) -> dict | None:
        for p in self.players:
            if p.get("playerId") == player_id:
                return p
        return None

    def find_node(self, node_id: str) -> dict | None:
        for n in self.nodes:
            if n.get("nodeId") == node_id:
                return n
        return None


class OverMessage:
    """Parsed 'over' message from the server."""

    def __init__(self, data: dict[str, Any]):
        self.raw = data
        self.match_id: str = data.get("matchId", "")
        self.over_round: int = data.get("overRound", 0)
        self.result_type: str = data.get("resultType", "")
        self.over_reason: str = data.get("overReason", "")
        self.winner_player_id: int | None = data.get("winnerPlayerId")
        self.players: list[dict] = data.get("players", [])


def parse_message(raw: dict[str, Any]) -> StartMessage | InquireMessage | OverMessage | dict:
    """Parse a raw message dict into the appropriate message type.

    Returns the raw dict unchanged for unknown msg_name values,
    preserving any unknown fields.
    """
    msg_name = raw.get("msg_name", "")
    msg_data = raw.get("msg_data", {})
    if msg_name == "start":
        return StartMessage(msg_data)
    elif msg_name == "inquire":
        return InquireMessage(msg_data)
    elif msg_name == "over":
        return OverMessage(msg_data)
    else:
        return raw