"""Self state: identifying whether the controlled player can make a move decision."""

from __future__ import annotations

from typing import Any


# Player states where MOVE/active actions are allowed
_CAN_ACT_STATES = {"IDLE"}

# States where limited actions are allowed (WAIT, MOVE to target, horse, rush_speed)
_LIMITED_ACT_STATES = {"MOVING", "WAITING"}

# Player states where no active action should be sent (just heartbeat)
_PASSIVE_STATES = {"PROCESSING", "VERIFYING", "FORCED_PASSING", "RESTING"}

RESOURCE_CLAIM_PRIORITY = [
    "FAST_HORSE",
    "SHORT_HORSE",
    "ICE_BOX",
    "OFFICIAL_PERMIT",
    "PASS_TOKEN",
    "INTEL",
    "BOAT_RIGHT",
]

TASK_PRIORITY = {
    "T01": (30, 3, 10.0),
    "T06": (30, 3, 10.0),
    "T02": (30, 4, 7.5),
    "T08": (30, 4, 7.5),
    "T11": (30, 4, 7.5),
    "T04": (30, 6, 5.0),
    "T12": (15, 5, 3.0),
    "T13": (15, 5, 3.0),
    "T14": (15, 5, 3.0),
}

TASK_SCORE_TARGET = 90
TASK_SCORE_STRETCH = 110
MAX_TASK_DETOUR_COST = 15
ICE_BOX_FRESHNESS_THRESHOLD = 72
RUSH_PROTECT_FRESHNESS = 50
INTEL_MAX_DISTANCE = 15


def get_team_id(player: dict) -> str:
    return player.get("teamId", "")


def get_task_template_id(task: dict) -> str:
    return task.get("taskTemplateId") or task.get("templateId", "")


def node_has_obstacle(node: dict) -> bool:
    return bool(node.get("hasObstacle") or node.get("obstacle") or node.get("blocked"))


def is_enemy_guard(guard: dict | None, my_team_id: str, my_player_id: int | None = None) -> bool:
    if not guard:
        return False
    if guard.get("active") is False:
        return False
    if guard.get("defense", 0) <= 0:
        return False
    owner_team = guard.get("ownerTeamId")
    if owner_team:
        return bool(my_team_id) and owner_team != my_team_id
    owner_player = guard.get("playerId")
    if owner_player is not None and my_player_id is not None:
        return owner_player != my_player_id
    return False


def guard_is_active(guard: dict | None) -> bool:
    if not guard:
        return False
    if guard.get("active") is False:
        return False
    return guard.get("defense", 0) > 0


def is_verify_process(process_type: str | None) -> bool:
    return process_type in ("VERIFY", "VERIFY_GATE")


def can_act(player: dict) -> bool:
    return player.get("state", "") in _CAN_ACT_STATES


def is_in_limited_state(player: dict) -> bool:
    return player.get("state", "") in _LIMITED_ACT_STATES


def can_move(player: dict) -> bool:
    return can_act(player)


def is_in_passive_state(player: dict) -> bool:
    return player.get("state", "") in _PASSIVE_STATES


def get_current_node_id(player: dict) -> str | None:
    return player.get("currentNodeId")


def get_next_node_id(player: dict) -> str | None:
    return player.get("nextNodeId")


def is_delivered(player: dict) -> bool:
    return player.get("delivered", False)


def is_retired(player: dict) -> bool:
    return player.get("retired", False)


def needs_processing(node: dict | None) -> bool:
    if node is None:
        return False
    process_type = node.get("processType")
    return bool(process_type) and not is_verify_process(process_type)


def is_verified(player: dict) -> bool:
    return player.get("verified", False)


def is_at_node(player: dict, node_id: str) -> bool:
    return player.get("currentNodeId") == node_id


def get_good_fruit(player: dict) -> int:
    return player.get("goodFruit", 0)


def get_bad_fruit(player: dict) -> int:
    return player.get("badFruit", 0)


def get_freshness(player: dict) -> float:
    return player.get("freshness", 0.0)


def get_player_resources(player: dict) -> dict[str, int]:
    return player.get("resources", {}) or {}


def has_resource(player: dict, resource_type: str) -> bool:
    return get_player_resources(player).get(resource_type, 0) > 0


def get_squad_count(player: dict) -> int:
    if "squadAvailable" in player:
        return int(player.get("squadAvailable", 0))
    return int(player.get("squadMembers", 0))


def get_action_points(player: dict) -> int:
    if "guardActionPoint" in player:
        return int(player.get("guardActionPoint", 0))
    return int(player.get("actionPoints", 0))


def get_task_score(player: dict) -> int:
    return player.get("taskScore", 0)


def find_available_resources(node: dict | None) -> list[tuple[str, int]]:
    if node is None:
        return []
    stock = node.get("resourceStock", {})
    available = [(rtype, count) for rtype, count in stock.items() if count > 0]
    priority_map = {rtype: i for i, rtype in enumerate(RESOURCE_CLAIM_PRIORITY)}
    available.sort(key=lambda x: priority_map.get(x[0], 999))
    return available


def _task_is_claimable(task: dict, node_id: str, player_id: int, graph_neighbors: list[str] | None = None) -> bool:
    task_node = task.get("nodeId", "")
    template_id = get_task_template_id(task)
    if template_id.startswith("T04"):
        if task_node != node_id and (not graph_neighbors or task_node not in graph_neighbors):
            return False
    elif task_node != node_id:
        return False

    if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
        return False

    protection = task.get("protectionPlayerId", 0)
    if protection not in (0, player_id):
        return False

    owner = task.get("ownerPlayerId", 0)
    return owner in (0, player_id)


def find_task_at_node(
    tasks: list[dict],
    node_id: str,
    player_id: int,
    graph_neighbors: list[str] | None = None,
) -> dict | None:
    candidates = []
    for task in tasks:
        if _task_is_claimable(task, node_id, player_id, graph_neighbors):
            candidates.append(task)

    if not candidates:
        return None

    def task_priority(t: dict) -> float:
        template_id = get_task_template_id(t)
        if template_id in TASK_PRIORITY:
            return -TASK_PRIORITY[template_id][2]
        for prefix in TASK_PRIORITY:
            if template_id.startswith(prefix):
                return -TASK_PRIORITY[prefix][2]
        return 0.0

    candidates.sort(key=task_priority)
    return candidates[0]


def get_blocked_nodes(
    inquire_nodes: list[dict],
    my_team_id: str,
    my_player_id: int | None = None,
) -> set[str]:
    blocked = set()
    for node in inquire_nodes:
        nid = node.get("nodeId", "")
        if not nid:
            continue
        if is_enemy_guard(node.get("guard"), my_team_id, my_player_id):
            blocked.add(nid)
        if node_has_obstacle(node):
            blocked.add(nid)
    return blocked


def classify_opponent_mode(
    my_player: dict,
    opp_player: dict | None,
    phase: str,
) -> str:
    if is_delivered(my_player):
        return "DELIVERED"

    if opp_player is None:
        return "STEADY"

    my_score = my_player.get("totalScore", 0)
    opp_score = opp_player.get("totalScore", 0)
    opp_task_score = opp_player.get("taskScore", 0)

    my_node = my_player.get("currentNodeId", "")
    opp_node = opp_player.get("currentNodeId", "")
    if my_node == opp_node and my_node and phase == "RUSH":
        return "GATE_FIGHT"

    if my_score > opp_score:
        if opp_task_score < TASK_SCORE_TARGET:
            return "STEADY"
        return "CONSERVATIVE"
    return "AGGRESSIVE"
