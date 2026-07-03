"""Decision construction: building protocol-legal action messages."""

from __future__ import annotations

from typing import Any


def make_registration(player_id: int, player_name: str, version: str = "1.0") -> dict:
    """Build a registration message.

    Protocol fields (from 通信协议 第4章):
    - msg_name: "registration"
    - msg_data.playerId: Int (必传)
    - msg_data.playerName: String (必传)
    - msg_data.version: String (必传)
    """
    return {
        "msg_name": "registration",
        "msg_data": {
            "playerId": player_id,
            "playerName": player_name,
            "version": version,
        }
    }


def make_ready(match_id: str, round_num: int, player_id: int) -> dict:
    """Build a ready message.

    Protocol fields (from 通信协议 第6章):
    - msg_name: "ready"
    - msg_data.matchId: String (必传, must equal start.matchId)
    - msg_data.round: Int (必传, must equal start.round)
    - msg_data.playerId: Int (必传)
    """
    return {
        "msg_name": "ready",
        "msg_data": {
            "matchId": match_id,
            "round": round_num,
            "playerId": player_id,
        }
    }


def make_action(match_id: str, round_num: int, player_id: int, actions: list[dict] | None = None) -> dict:
    """Build an action message.

    Protocol fields (from 通信协议 第8章):
    - msg_name: "action"
    - msg_data.matchId: String (必传)
    - msg_data.round: Int (必传, must equal inquire.round)
    - msg_data.playerId: Int (必传)
    - msg_data.actions: Array (必传, empty array for heartbeat)
    """
    return {
        "msg_name": "action",
        "msg_data": {
            "matchId": match_id,
            "round": round_num,
            "playerId": player_id,
            "actions": actions if actions is not None else [],
        }
    }


def make_move_action(target_node_id: str) -> dict:
    """Build a MOVE action item.

    Protocol fields (from 通信协议 第8章 actions[] 动作字段矩阵):
    - action: "MOVE" (必填)
    - targetNodeId: String (必填)
    """
    return {
        "action": "MOVE",
        "targetNodeId": target_node_id,
    }


def make_wait_action() -> dict:
    """Build a WAIT action item.

    Protocol fields (from 通信协议 第8章 actions[] 动作字段矩阵):
    - action: "WAIT" (必填)
    """
    return {
        "action": "WAIT",
    }


def make_process_action(target_node_id: str | None = None) -> dict:
    """Build a PROCESS action item.

    Protocol fields (from 通信协议 第8章):
    - action: "PROCESS" (必填)
    - targetNodeId: String (可选)
    """
    action = {"action": "PROCESS"}
    if target_node_id is not None:
        action["targetNodeId"] = target_node_id
    return action


def make_dock_action(target_node_id: str | None = None) -> dict:
    """Build a DOCK action item.

    Protocol fields (from 通信协议 第8章):
    - action: "DOCK" (必填)
    - targetNodeId: String (可选)
    """
    action = {"action": "DOCK"}
    if target_node_id is not None:
        action["targetNodeId"] = target_node_id
    return action


def make_verify_gate_action(target_node_id: str | None = None) -> dict:
    """Build a VERIFY_GATE action item.

    Protocol fields (from 通信协议 第8章):
    - action: "VERIFY_GATE" (必填)
    - targetNodeId: String (可选)
    """
    action = {"action": "VERIFY_GATE"}
    if target_node_id is not None:
        action["targetNodeId"] = target_node_id
    return action


def make_window_card_action(contest_id: str, card: str = "ABSTAIN") -> dict:
    """Build a WINDOW_CARD action item.

    Protocol fields (from 通信协议 第8章):
    - action: "WINDOW_CARD" (必填)
    - contestId: String (必填)
    - card: String (必填, e.g. "ABSTAIN")
    """
    return {
        "action": "WINDOW_CARD",
        "contestId": contest_id,
        "card": card,
    }


def make_empty_action(match_id: str, round_num: int, player_id: int) -> dict:
    """Build an action message with empty actions (safe heartbeat).

    This is the safe fallback when MOVE is not possible.
    """
    return make_action(match_id, round_num, player_id, actions=[])


def make_claim_resource_action(target_node_id: str, resource_type: str) -> dict:
    """Build a CLAIM_RESOURCE action item.

    Protocol fields (from 通信协议 第8章):
    - action: "CLAIM_RESOURCE" (必填)
    - targetNodeId: String (必填)
    - resourceType: String (必填)
    """
    return {
        "action": "CLAIM_RESOURCE",
        "targetNodeId": target_node_id,
        "resourceType": resource_type,
    }


def make_claim_task_action(task_id: str) -> dict:
    """Build a CLAIM_TASK action item.

    Protocol fields (from 通信协议 第8章):
    - action: "CLAIM_TASK" (必填)
    - taskId: String (必填)
    """
    return {
        "action": "CLAIM_TASK",
        "taskId": task_id,
    }


def make_deliver_action() -> dict:
    """Build a DELIVER action item.

    Protocol fields (from 通信协议 第8章):
    - action: "DELIVER" (必填)
    """
    return {
        "action": "DELIVER",
    }


def make_break_guard_action(target_node_id: str, good_fruit: int = 0, bad_fruit: int = 0) -> dict:
    """Build a BREAK_GUARD action item.

    Protocol fields (from 通信协议 第8章):
    - action: "BREAK_GUARD" (必填)
    - targetNodeId: String (必填)
    - goodFruit: Int (可选, 0-2)
    - badFruit: Int (可选, 0-2)
    """
    action = {
        "action": "BREAK_GUARD",
        "targetNodeId": target_node_id,
        "goodFruit": good_fruit,
        "badFruit": bad_fruit,
    }
    return action


def make_forced_pass_action(target_node_id: str) -> dict:
    """Build a FORCED_PASS action item.

    Protocol fields (from 通信协议 第8章):
    - action: "FORCED_PASS" (必填)
    - targetNodeId: String (必填)
    """
    return {
        "action": "FORCED_PASS",
        "targetNodeId": target_node_id,
    }


def make_clear_action(target_node_id: str) -> dict:
    """Build a CLEAR action item.

    Protocol fields (from 通信协议 第8章):
    - action: "CLEAR" (必填)
    - targetNodeId: String (必填)
    """
    return {
        "action": "CLEAR",
        "targetNodeId": target_node_id,
    }


def make_set_guard_action(target_node_id: str, extra_good_fruit: int = 0) -> dict:
    """Build a SET_GUARD action item.

    Protocol fields (from 通信协议 第8章):
    - action: "SET_GUARD" (必填)
    - targetNodeId: String (必填)
    - extraGoodFruit: Int (可选, 0-2)
    """
    return {
        "action": "SET_GUARD",
        "targetNodeId": target_node_id,
        "extraGoodFruit": extra_good_fruit,
    }


def make_use_resource_action(resource_type: str, target_node_id: str | None = None) -> dict:
    """Build a USE_RESOURCE action item.

    Protocol fields (from 通信协议 第8章):
    - action: "USE_RESOURCE" (必填)
    - resourceType: String (必填)
    - targetNodeId: String (可选)
    """
    action = {
        "action": "USE_RESOURCE",
        "resourceType": resource_type,
    }
    if target_node_id is not None:
        action["targetNodeId"] = target_node_id
    return action


def make_squad_scout_action(target_node_id: str) -> dict:
    """Build a SQUAD_SCOUT action item."""
    return {"action": "SQUAD_SCOUT", "targetNodeId": target_node_id}


def make_squad_clear_action(target_node_id: str) -> dict:
    """Build a SQUAD_CLEAR action item."""
    return {"action": "SQUAD_CLEAR", "targetNodeId": target_node_id}


def make_squad_reinforce_action(target_node_id: str) -> dict:
    """Build a SQUAD_REINFORCE action item."""
    return {"action": "SQUAD_REINFORCE", "targetNodeId": target_node_id}


def make_squad_weaken_action(target_node_id: str) -> dict:
    """Build a SQUAD_WEAKEN action item."""
    return {"action": "SQUAD_WEAKEN", "targetNodeId": target_node_id}


def make_rush_speed_action() -> dict:
    """Build a RUSH_SPEED action item (疾行令)."""
    return {"action": "RUSH_SPEED"}


def make_rush_protect_action() -> dict:
    """Build a RUSH_PROTECT action item (护果令)."""
    return {"action": "RUSH_PROTECT"}