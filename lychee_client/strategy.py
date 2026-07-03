"""Strategy: full decision engine implementing 策略设计文档 L2–L8.

Priority order per frame (策略文档 §14 伪代码):
  P0  Stable online, every-frame heartbeat, zero illegal actions
  P1  Must deliver (goodFruit>0, freshness>0, verified, at S15)
  P2  Task score ≥ 90
  P3  Deliver early (time score)
  P4  Preserve good fruit & freshness
  P5  Moderate combat (guard/break/squad) without sacrificing P1–P4
"""

from __future__ import annotations

import logging
from typing import Any

from lychee_client.map_graph import MapGraph, ROUTE_FRESHNESS_LOSS
from lychee_client.planner import (
    GlobalPlan,
    MapProfile,
    MIN_RESOURCE_NET_VALUE,
    MIN_TASK_NET_VALUE,
    build_global_plan,
    estimate_delivery_route,
    resource_net_value,
    should_set_guard_now,
    task_net_value,
)
from lychee_client.state import (
    can_move, can_act, get_current_node_id, needs_processing,
    is_delivered, is_retired, is_verified, is_at_node, is_in_passive_state,
    is_in_limited_state,
    find_available_resources, find_task_at_node,
    get_good_fruit, get_bad_fruit, get_freshness,
    get_player_resources, has_resource, get_squad_count,
    get_action_points, get_task_score, get_blocked_nodes,
    classify_opponent_mode, get_team_id, get_task_template_id,
    is_verify_process, is_enemy_guard, guard_is_active, node_has_obstacle,
    TASK_SCORE_TARGET, TASK_SCORE_STRETCH, MAX_TASK_DETOUR_COST,
    ICE_BOX_FRESHNESS_THRESHOLD, RUSH_PROTECT_FRESHNESS,
    RESOURCE_CLAIM_PRIORITY, TASK_PRIORITY,
)
from lychee_client.decision import (
    make_action, make_move_action, make_wait_action,
    make_process_action, make_dock_action, make_verify_gate_action,
    make_empty_action, make_window_card_action,
    make_claim_resource_action, make_claim_task_action,
    make_deliver_action, make_break_guard_action,
    make_forced_pass_action, make_clear_action, make_set_guard_action,
    make_use_resource_action,
    make_squad_scout_action, make_squad_clear_action,
    make_squad_reinforce_action, make_squad_weaken_action,
    make_rush_protect_action,
)

logger = logging.getLogger("lychee_client.strategy")


def _make_process_action(
    match_id: str,
    round_num: int,
    player_id: int,
    process_type: str,
    current_node_id: str,
    phase: str,
) -> dict:
    """Map processType to the correct protocol action."""
    if process_type == "DOCK":
        return make_action(match_id, round_num, player_id, [make_dock_action(current_node_id)])
    if is_verify_process(process_type):
        if phase == "RUSH":
            return make_action(match_id, round_num, player_id, [make_verify_gate_action(current_node_id)])
        return make_empty_action(match_id, round_num, player_id)
    return make_action(match_id, round_num, player_id, [make_process_action(current_node_id)])


def _append_squad_action(
    action_msg: dict,
    squad_action: dict | None,
) -> dict:
    if squad_action is None:
        return action_msg
    actions = action_msg.get("msg_data", {}).get("actions", [])
    if len(actions) >= 2:
        return action_msg
    if len(actions) == 1:
        actions = actions + [squad_action]
    else:
        actions = [squad_action]
    action_msg["msg_data"]["actions"] = actions
    return action_msg


def decide_action(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    contests: list[dict] | None = None,
    events: list[dict] | None = None,
    active_contest_id: str = "",
    last_move_failed: bool = False,
    last_move_error: str = "",
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    phase: str = "",
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    weather: dict | None = None,
    all_players: list[dict] | None = None,
    inquire_nodes: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    pending_task_hold_task_id: str = "",
    pending_task_hold_node_id: str = "",
    pending_task_hold_until_round: int = 0,
    forced_pass_failed_targets: set[str] | None = None,
    map_profile: MapProfile | None = None,
) -> dict:
    """Decide the action for the current round.

    Implements the single-frame decision pseudocode from 策略文档 §14.
    Returns a complete action message dict.
    """
    # Defaults
    if terminal_node_ids is None:
        terminal_node_ids = []
    if tasks is None:
        tasks = []
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if weather is None:
        weather = {}
    if all_players is None:
        all_players = []
    if inquire_nodes is None:
        inquire_nodes = []
    if failed_task_ids is None:
        failed_task_ids = set()
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if forced_pass_failed_targets is None:
        forced_pass_failed_targets = set()

    try:
        return _decide_action_impl(
            match_id, round_num, player_id, player, graph,
            current_node, process_nodes, contests, events,
            active_contest_id, last_move_failed, last_move_error,
            gate_node_id, terminal_node_ids, tasks, phase,
            processed_node_ids, visited_node_ids, weather, all_players, inquire_nodes,
            failed_task_ids, rush_speed_failed, guard_blocked_targets, avoid_route_nodes,
            pending_task_hold_task_id, pending_task_hold_node_id, pending_task_hold_until_round,
            forced_pass_failed_targets, map_profile,
        )
    except Exception as e:
        logger.error("Round %d: Strategy error: %s", round_num, e, exc_info=True)
        return make_empty_action(match_id, round_num, player_id)


def _decide_action_impl(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    contests: list[dict] | None = None,
    events: list[dict] | None = None,
    active_contest_id: str = "",
    last_move_failed: bool = False,
    last_move_error: str = "",
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    phase: str = "",
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    weather: dict | None = None,
    all_players: list[dict] | None = None,
    inquire_nodes: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    pending_task_hold_task_id: str = "",
    pending_task_hold_node_id: str = "",
    pending_task_hold_until_round: int = 0,
    forced_pass_failed_targets: set[str] | None = None,
    map_profile: MapProfile | None = None,
) -> dict:
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if forced_pass_failed_targets is None:
        forced_pass_failed_targets = set()

    # --- P0: Stability ---
    if is_retired(player) or is_delivered(player):
        return make_empty_action(match_id, round_num, player_id)

    state = player.get("state", "")
    current_node_id = get_current_node_id(player)
    my_team_id = get_team_id(player)

    # If in CONTESTING state, we must send WINDOW_CARD
    if state == "CONTESTING":
        on_water_route = _is_on_water_route(graph, current_node_id, gate_node_id, terminal_node_ids)
        return _handle_contesting(
            match_id, round_num, player_id, player,
            contests, events, active_contest_id, player,
            all_players, phase, on_water_route,
        )

    # Passive states: PROCESSING, VERIFYING, FORCED_PASSING, RESTING → heartbeat
    if is_in_passive_state(player):
        return make_empty_action(match_id, round_num, player_id)

    blocked = get_blocked_nodes(inquire_nodes, my_team_id, player_id)
    route_blocked = set(blocked)
    route_blocked.update(guard_blocked_targets)
    route_blocked.update(avoid_route_nodes)
    opp_player = _find_opponent(all_players, player_id)
    mode = classify_opponent_mode(player, opp_player, phase)

    obstacle_nodes: set[str] = set()
    for node in inquire_nodes:
        if node_has_obstacle(node):
            obstacle_nodes.add(node.get("nodeId", ""))
    global_plan = build_global_plan(
        round_num, player, graph, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids, route_blocked,
        obstacle_nodes, all_players, player_id, phase, map_profile,
    )
    force_delivery = _should_force_delivery(round_num, phase, player, global_plan)

    if is_in_limited_state(player):
        guard_target = _resolve_guard_block_target(player, route_blocked, guard_blocked_targets)

        if state == "WAITING":
            next_node = player.get("nextNodeId", "")
            if last_move_failed and last_move_error == "OBJECT_BUSY":
                logger.info("Round %d: OBJECT_BUSY in WAITING, sending WAIT", round_num)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            pending_process_type = _get_pending_station_process_type(
                current_node_id, next_node, process_nodes, processed_node_ids,
            )
            if pending_process_type:
                if _has_current_process_for_node(player, current_node_id):
                    logger.info("Round %d: station process running at %s, sending empty action", round_num, current_node_id)
                    return make_empty_action(match_id, round_num, player_id)
                logger.info("Round %d: station process not started at %s, retrying %s", round_num, current_node_id, pending_process_type)
                return _make_process_action(
                    match_id, round_num, player_id,
                    pending_process_type, current_node_id, phase,
                )

            if last_move_failed and last_move_error == "PROCESS_REQUIRED":
                process_type = process_nodes.get(current_node_id, {}).get("processType") if process_nodes and current_node_id else ""
                if process_type:
                    logger.info("Round %d: PROCESS_REQUIRED in WAITING at %s, retrying %s", round_num, current_node_id, process_type)
                    return _make_process_action(
                        match_id, round_num, player_id,
                        process_type, current_node_id, phase,
                    )
                logger.info("Round %d: PROCESS_REQUIRED in WAITING at %s, sending WAIT", round_num, current_node_id)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            if not force_delivery and current_node_id and not next_node:
                if (
                    pending_task_hold_node_id == current_node_id
                    and round_num <= pending_task_hold_until_round
                ):
                    logger.info(
                        "Round %d: waiting for busy task at %s until %d",
                        round_num, current_node_id, pending_task_hold_until_round,
                    )
                    return make_action(match_id, round_num, player_id, [make_wait_action()])
                if pending_task_hold_node_id == current_node_id and pending_task_hold_task_id:
                    task_retry = _retry_task_at_current_node(
                        match_id, round_num, player_id, player, graph,
                        current_node_id, tasks, failed_task_ids,
                        preferred_task_id=pending_task_hold_task_id,
                    )
                    if task_retry is not None:
                        return task_retry
                task_retry = _retry_task_at_current_node(
                    match_id, round_num, player_id, player, graph,
                    current_node_id, tasks, failed_task_ids,
                )
                if task_retry is not None:
                    return task_retry

            if force_delivery and current_node_id and not next_node:
                direct_target = _find_direct_delivery_step(
                    graph, current_node_id, player, gate_node_id, terminal_node_ids,
                    weather, process_nodes, processed_node_ids,
                )
                if direct_target:
                    if direct_target in route_blocked or direct_target in obstacle_nodes:
                        # WAITING can only WAIT/MOVE (任务书 §8.2): cannot break the
                        # guard here. Weaken via squad and retry MOVE; the guard is
                        # broken decisively once we fall back to IDLE.
                        return _move_and_weaken_guard(
                            match_id, round_num, player_id, player,
                            inquire_nodes, direct_target, my_team_id,
                        )
                    logger.info("Round %d: FORCE_DELIVERY move to %s (WAITING)", round_num, direct_target)
                    return make_action(match_id, round_num, player_id, [make_move_action(direct_target)])

            if guard_target:
                if next_node and guard_target == next_node:
                    return _move_and_weaken_guard(
                        match_id, round_num, player_id, player,
                        inquire_nodes, next_node, my_team_id,
                    )
                return _wait_and_weaken_guard(
                    match_id, round_num, player_id, player,
                    inquire_nodes, guard_target, my_team_id,
                )

            if last_move_failed and last_move_error in ("OBJECT_BUSY", "MOVING_ACTION_FORBIDDEN"):
                logger.info("Round %d: %s in WAITING, sending WAIT", round_num, last_move_error)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            if next_node:
                if next_node in route_blocked:
                    return _move_and_weaken_guard(
                        match_id, round_num, player_id, player,
                        inquire_nodes, next_node, my_team_id,
                    )
                return make_action(match_id, round_num, player_id, [make_move_action(next_node)])
            if current_node_id:
                move_target = _find_move_target(
                    graph, current_node_id, player, gate_node_id, terminal_node_ids,
                    weather, route_blocked, obstacle_nodes=obstacle_nodes,
                    process_nodes=process_nodes,
                    processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
                )
                if move_target and move_target not in route_blocked:
                    return make_action(match_id, round_num, player_id, [make_move_action(move_target)])
                if move_target:
                    # Blocked exit in WAITING: only MOVE/WAIT legal, weaken via squad.
                    return _move_and_weaken_guard(
                        match_id, round_num, player_id, player,
                        inquire_nodes, move_target, my_team_id,
                    )

        if state == "MOVING":
            if guard_target or last_move_failed and last_move_error == "MOVE_BLOCKED_BY_GUARD":
                target = guard_target or player.get("nextNodeId", "")
                return _continue_and_weaken_guard(
                    match_id, round_num, player_id, player,
                    inquire_nodes, target, my_team_id,
                )
            moving_action = _handle_moving(match_id, round_num, player_id, player, graph, weather, phase)
            if moving_action.get("msg_data", {}).get("actions"):
                return moving_action

        return make_empty_action(match_id, round_num, player_id)

    # Must be IDLE to act
    if not can_act(player):
        return make_empty_action(match_id, round_num, player_id)

    if current_node_id is None:
        return make_empty_action(match_id, round_num, player_id)

    if (
        not force_delivery
        and pending_task_hold_node_id == current_node_id
        and round_num <= pending_task_hold_until_round
    ):
        logger.info(
            "Round %d: holding at %s for busy task until %d",
            round_num, current_node_id, pending_task_hold_until_round,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])
    if (
        not force_delivery
        and pending_task_hold_node_id == current_node_id
        and pending_task_hold_task_id
    ):
        task_retry = _retry_task_at_current_node(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, failed_task_ids,
            preferred_task_id=pending_task_hold_task_id,
        )
        if task_retry is not None:
            return task_retry

    # Don't use blocked_nodes as hard filter in BFS — it causes TARGET_NOT_REACHABLE
    # Instead, use weighted routing to prefer unblocked paths
    blocked_soft = route_blocked  # used for weighted routing and combat

    # --- P1: Delivery flow (策略文档 §4.2 FSM) ---

    # At S15: DELIVER if verified and can deliver
    if current_node_id in terminal_node_ids:
        if is_verified(player) and get_good_fruit(player) > 0 and get_freshness(player) > 0:
            return make_action(match_id, round_num, player_id, [make_deliver_action()])
        # At S15 but not verified → go back to S14 (策略文档 §4.2: 无视设卡与障碍)
        if gate_node_id and not is_verified(player):
            # Direct move to S14 — guards/obstacles don't block this path
            step = graph.next_step_toward(current_node_id, gate_node_id, weather, None)
            if step:
                return make_action(match_id, round_num, player_id, [make_move_action(step)])
        # At S15, verified but no good fruit/freshness → WAIT (can't deliver)
        return make_empty_action(match_id, round_num, player_id)

    # At S14 (gate): VERIFY_GATE in RUSH phase
    if gate_node_id and is_at_node(player, gate_node_id) and not is_verified(player):
        if phase == "RUSH":
            action = make_verify_gate_action(current_node_id)
            if get_bad_fruit(player) >= 2 or get_good_fruit(player) >= 1:
                action["rushTactic"] = "BREAK_ORDER"
                logger.info("Round %d: VERIFY_GATE with BREAK_ORDER at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [action])
        # Not RUSH yet: don't submit VERIFY_GATE (will be rejected)
        # Continue doing other things until RUSH

    # --- Fixed processing (策略文档 §4.1: 再次到达同一站需重新处理) ---
    # Process at current node ONLY if not already processed this visit.
    # processed_node_ids tracks nodes where we completed processing this session.
    # If already processed, skip to MOVE (even if node has processType).
    already_processed_here = current_node_id in processed_node_ids
    process_type = None if already_processed_here else _get_process_type(current_node, process_nodes, current_node_id)

    if process_type:
        if last_move_failed and "WINDOW" in last_move_error.upper():
            move_target = _find_move_target(
                graph, current_node_id, player, gate_node_id, terminal_node_ids,
                weather, route_blocked, obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
                processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
            )
            if move_target:
                return make_action(match_id, round_num, player_id, [make_move_action(move_target)])
            return make_empty_action(match_id, round_num, player_id)

        return _make_process_action(match_id, round_num, player_id, process_type, current_node_id, phase)

    if last_move_failed and last_move_error == "PROCESS_REQUIRED":
        process_type = _get_process_type(current_node, process_nodes, current_node_id)
        if process_type:
            logger.info("Round %d: PROCESS_REQUIRED at %s, sending %s", round_num, current_node_id, process_type)
            return _make_process_action(match_id, round_num, player_id, process_type, current_node_id, phase)
        return make_action(match_id, round_num, player_id, [make_process_action(current_node_id)])

    # --- Handle OBJECT_BUSY: wait one round and retry ---
    if last_move_failed and last_move_error == "OBJECT_BUSY":
        # The process target is busy (e.g., window contest just ended, still transitioning)
        # Wait one round, then retry process on next round
        logger.info("Round %d: OBJECT_BUSY, waiting", round_num)
        return make_action(match_id, round_num, player_id, [make_wait_action()])

    # --- Handle blocked movement ---
    if last_move_failed and last_move_error == "MOVE_BLOCKED_BY_GUARD":
        return _handle_blocked_by_guard(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked_soft, inquire_nodes, process_nodes=process_nodes,
        )

    # --- P2/P3: Task strategy (策略文档 §5) ---
    if not force_delivery:
        task_action = _handle_tasks(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, player_id, phase, weather, blocked,
            goal_node_id=gate_node_id, terminal_node_ids=terminal_node_ids,
            obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
            processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
            failed_task_ids=failed_task_ids,
            global_plan=global_plan,
        )
        if task_action is not None:
            return task_action

    # --- P4: Resource strategy (策略文档 §6) ---
    # Skip resource claiming when close to gate (prioritize delivery)
    dist_to_gate = 0
    if gate_node_id:
        dist_to_gate = graph.path_length(current_node_id, gate_node_id, weather, None)
    if not force_delivery and dist_to_gate > 4:  # Only claim resources when not close to gate
        resource_action = _handle_resources(
            match_id, round_num, player_id, player, graph,
            current_node_id, current_node, phase, weather,
            global_plan=global_plan,
            gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids,
            process_nodes=process_nodes,
            processed_node_ids=processed_node_ids,
            inquire_nodes=inquire_nodes,
            map_profile=map_profile,
        )
        if resource_action is not None:
            return resource_action
    if force_delivery:
        resource_action = _handle_force_delivery_resource(
            match_id, round_num, player_id, player, graph,
            current_node_id, current_node, gate_node_id,
            terminal_node_ids, weather, process_nodes, processed_node_ids,
            global_plan=global_plan,
            map_profile=map_profile,
        )
        if resource_action is not None:
            return resource_action

    # --- P5: Use resources (ice box, horses) ---
    use_res_action = _handle_use_resources(
        match_id, round_num, player_id, player,
        current_node_id, graph, weather, phase,
        global_plan=global_plan,
    )
    if use_res_action is not None:
        return use_res_action

    # --- P5: Combat (策略文档 §8) — guard, break, squad ---
    if not force_delivery:
        combat_action = _handle_combat(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked_soft, mode, phase, inquire_nodes, opp_player,
            obstacle_nodes=obstacle_nodes,
            process_nodes=process_nodes,
            visited_node_ids=visited_node_ids,
            my_team_id=my_team_id,
            global_plan=global_plan,
        )
        if combat_action is not None:
            return combat_action

    # --- Rush tactics (策略文档 §10) ---
    rush_action = _handle_rush_tactics(
        match_id, round_num, player_id, player,
        current_node_id, phase, mode,
        graph=graph, gate_node_id=gate_node_id,
        terminal_node_ids=terminal_node_ids, weather=weather,
        obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
        rush_speed_failed=rush_speed_failed,
    )
    if rush_action is not None:
        return rush_action

    # --- NAVIGATION: Move toward goal ---
    if force_delivery:
        direct_target = _find_direct_delivery_step(
            graph, current_node_id, player, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids,
        )
        if direct_target:
            if direct_target in route_blocked or direct_target in obstacle_nodes:
                blocker_action = _handle_force_delivery_blocker(
                    match_id, round_num, player_id, player,
                    direct_target, inquire_nodes, tasks, failed_task_ids,
                    obstacle_nodes, my_team_id,
                )
                if blocker_action.get("msg_data", {}).get("actions"):
                    return blocker_action
            logger.info("Round %d: FORCE_DELIVERY move to %s (goal=%s)", round_num, direct_target, gate_node_id)
            return make_action(match_id, round_num, player_id, [make_move_action(direct_target)])

    move_target = _find_move_target(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, route_blocked, obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids,
        visited_node_ids=set() if force_delivery else visited_node_ids,
    )

    # Next hop has enemy guard → break / forced pass / detour before MOVE
    if move_target and move_target in route_blocked:
        guard_action = _handle_blocked_by_guard(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, route_blocked, inquire_nodes, process_nodes=process_nodes,
        )
        if guard_action.get("msg_data", {}).get("actions"):
            return guard_action

    # Handle obstacle on next step (策略文档 §3.4: 道路障碍 → T04/CLEAR/FORCED_PASS)
    if move_target and move_target in obstacle_nodes:
        # Priority 1: CLAIM_TASK if T04 task exists at obstacle node (score + clear)
        t04_task = None
        for task in tasks:
            if (task.get("nodeId") == move_target
                    and task.get("active", False)
                    and not task.get("completed", False)
                    and not task.get("failed", False)
                    and get_task_template_id(task).startswith("T04")
                    and task.get("taskId", "") not in failed_task_ids):
                t04_task = task
                break
        if t04_task:
            logger.info("Round %d: Obstacle at %s, claiming T04 task", round_num, move_target)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(t04_task.get("taskId", ""))
            ])

        # Priority 2: CLEAR if we have good fruit to spare (策略文档 §3.4: 1好果6帧)
        good_fruit = get_good_fruit(player)
        if good_fruit >= 2:  # Reserve at least 1 for DELIVER
            logger.info("Round %d: Obstacle at %s, using CLEAR", round_num, move_target)
            return make_action(match_id, round_num, player_id, [
                make_clear_action(move_target)
            ])

        # Priority 3: FORCED_PASS for obstacle-only (策略文档 §3.4: 8帧, no fruit cost)
        logger.info("Round %d: Obstacle at %s, using FORCED_PASS", round_num, move_target)
        return make_action(match_id, round_num, player_id, [
            make_forced_pass_action(move_target)
        ])
    if move_target:
        logger.info("Round %d: NAV move to %s (goal=%s)", round_num, move_target,
                     gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "?"))
        return make_action(match_id, round_num, player_id, [make_move_action(move_target)])

    return make_empty_action(match_id, round_num, player_id)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_opponent(all_players: list[dict], my_player_id: int) -> dict | None:
    """Find the opponent player dict."""
    for p in all_players:
        if p.get("playerId") != my_player_id:
            return p
    return None


def _is_on_water_route(
    graph: MapGraph, current_node_id: str,
    gate_node_id: str, terminal_node_ids: list[str],
) -> bool:
    """Check if the current path to goal goes through water edges."""
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if not goal or not current_node_id:
        return False
    path = graph.weighted_shortest_path(current_node_id, goal)
    if not path:
        return False
    for i in range(len(path) - 1):
        if graph.get_edge_route_type(path[i], path[i + 1]) == "WATER":
            return True
    return False


def _get_process_type(
    current_node: dict | None,
    process_nodes: dict[str, dict] | None,
    current_node_id: str,
) -> str | None:
    """Get the process type for the current node."""
    if current_node and needs_processing(current_node):
        return current_node.get("processType", "")
    if process_nodes and current_node_id in process_nodes:
        pn = process_nodes[current_node_id]
        return pn.get("processType", "")
    return None


def _get_goal_node(
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    graph: MapGraph,
    current_node_id: str,
    weather: dict | None = None,
    blocked: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
) -> str | None:
    """Determine the current navigation goal based on player state."""
    if is_delivered(player):
        return None
    if not is_verified(player) and gate_node_id:
        return gate_node_id
    if is_verified(player) and terminal_node_ids:
        # Find nearest terminal via weighted path
        best = None
        best_cost = float('inf')
        for tid in terminal_node_ids:
            path = graph.weighted_shortest_path(current_node_id, tid, weather, blocked, process_nodes)
            if path:
                cost = sum(graph.edge_cost(path[i], path[i+1], weather, blocked, process_nodes)
                           for i in range(len(path)-1))
                if cost < best_cost:
                    best_cost = cost
                    best = tid
        return best
    return None


def _should_force_delivery(
    round_num: int,
    phase: str,
    player: dict,
    global_plan: GlobalPlan | None = None,
) -> bool:
    """Stop optional scoring once delivery risk is higher than task/resource value."""
    if global_plan is not None:
        return global_plan.should_force_delivery
    if phase == "RUSH":
        return True
    if round_num >= 95 and get_task_score(player) >= 60:
        return True
    if round_num >= 175:
        return True
    return False


def _find_direct_delivery_step(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> str | None:
    goal_node = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, None, process_nodes,
    )
    if not goal_node:
        return None

    remaining_process_nodes = None
    if process_nodes:
        remaining_process_nodes = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    # Ignore guards/obstacles here. If the direct next hop is blocked, handle
    # that blocker explicitly instead of oscillating through detours.
    step = graph.next_step_toward(
        current_node_id, goal_node, weather, None,
        use_weighted=True, process_nodes=remaining_process_nodes,
    )
    if step:
        return step
    return graph.next_step_toward(current_node_id, goal_node, weather, None, use_weighted=False)


def _handle_force_delivery_blocker(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    target_node_id: str,
    inquire_nodes: list[dict],
    tasks: list[dict],
    failed_task_ids: set[str],
    obstacle_nodes: set[str],
    my_team_id: str,
) -> dict:
    # NOTE: only called from IDLE-equivalent states. BREAK_GUARD / CLEAR /
    # FORCED_PASS are illegal while MOVING/WAITING (任务书 §8.2), so callers must
    # not route WAITING/MOVING here.

    # Priority 1: an enemy guard blocks movement even after the obstacle is
    # cleared, and CLEAR / T04 do NOT remove a guard. A guard can only be broken
    # while IDLE, so resolve it first and decisively (repeat BREAK_GUARD until the
    # defense hits 0), speeding it up with a parallel squad weaken.
    for node in inquire_nodes:
        if node.get("nodeId") != target_node_id:
            continue
        guard = node.get("guard", {})
        if is_enemy_guard(guard, my_team_id, player_id):
            good = min(get_good_fruit(player), 2)
            bad = min(get_bad_fruit(player), 2)
            if good + bad > 0:
                logger.info(
                    "Round %d: FORCE_DELIVERY break guard at %s (gf=%d bf=%d)",
                    round_num, target_node_id, good, bad,
                )
                msg = make_action(match_id, round_num, player_id, [
                    make_break_guard_action(target_node_id, good_fruit=good, bad_fruit=bad)
                ])
                squad = _make_squad_weaken_action(
                    inquire_nodes, target_node_id, my_team_id, player_id, player,
                )
                return _append_squad_action(msg, squad) if squad else msg
            logger.info("Round %d: FORCE_DELIVERY forced pass guard at %s (no fruit)", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])
        break

    # Priority 2: a road-only obstacle (no enemy guard). Prefer T04 if it scores,
    # otherwise FORCED_PASS: it is a deterministic fixed tax (任务书 §6.3.2) and
    # avoids the CLEAR path that can stall on an OBJECT_BUSY obstacle window.
    if target_node_id in obstacle_nodes:
        for task in tasks:
            if (task.get("nodeId") == target_node_id
                    and task.get("active", False)
                    and not task.get("completed", False)
                    and not task.get("failed", False)
                    and get_task_template_id(task).startswith("T04")
                    and task.get("taskId", "") not in failed_task_ids):
                logger.info("Round %d: FORCE_DELIVERY T04 clear at %s", round_num, target_node_id)
                return make_action(match_id, round_num, player_id, [
                    make_claim_task_action(task.get("taskId", ""))
                ])
        logger.info("Round %d: FORCE_DELIVERY forced pass obstacle at %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])

    logger.info("Round %d: FORCE_DELIVERY forced pass blocked %s", round_num, target_node_id)
    return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])


def _find_move_target(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None = None,
    blocked: set[str] | None = None,
    failed_target: str = "",
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
) -> str | None:
    """Find the best move target using weighted shortest path toward the current goal.

    Filters out obstacle nodes (hasObstacle=true) from move targets,
    since MOVE to an obstacle node will be rejected with TARGET_NOT_REACHABLE.
    """
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()

    neighbors = graph.get_neighbors(current_node_id)
    if not neighbors:
        return None

    # Filter out failed target, obstacle nodes, and enemy-guarded nodes when alternatives exist
    guard_blocked = blocked or set()
    available = [n for n in neighbors if n != failed_target and n not in obstacle_nodes]
    if guard_blocked:
        safe = [n for n in available if n not in guard_blocked]
        if safe:
            available = safe
    forward_available = [n for n in available if n not in visited_node_ids]
    if forward_available:
        available = forward_available
    logger.info("_find_move_target: current=%s neighbors=%s available=%s visited=%s failed_target=%s",
                current_node_id, neighbors, available, visited_node_ids, failed_target)
    if not available:
        # Fall back: allow backtrack but still avoid guarded nodes if possible
        available = [n for n in neighbors if n != failed_target and n not in obstacle_nodes]
        if guard_blocked:
            safe = [n for n in available if n not in guard_blocked]
            if safe:
                available = safe
        if not available:
            available = neighbors

    goal_node = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current_node_id, weather, None, process_nodes)

    # Build remaining process nodes (exclude already-processed nodes at current visit)
    remaining_process_nodes = None
    if process_nodes:
        remaining_process_nodes = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    if goal_node:
        # Build soft-blocked set: obstacles + visited + enemy guards
        soft_blocked = set(obstacle_nodes)
        soft_blocked.update(visited_node_ids)
        soft_blocked.update(guard_blocked)
        # Don't block the goal itself
        soft_blocked.discard(goal_node)

        # Use weighted Dijkstra first — prefers WATER(1250) over ROAD(1380) over MOUNTAIN(1780)
        step = graph.next_step_toward(current_node_id, goal_node, weather, soft_blocked, use_weighted=True, process_nodes=remaining_process_nodes)
        if step and step in available:
            return step
        # Fallback: unweighted BFS
        step = graph.next_step_toward(current_node_id, goal_node, weather, soft_blocked, use_weighted=False)
        if step and step in available:
            return step
        # Try weighted without soft-blocked (just obstacles)
        step = graph.next_step_toward(current_node_id, goal_node, weather, obstacle_nodes, use_weighted=True, process_nodes=remaining_process_nodes)
        if step and step in available:
            return step
        # Try BFS without any filter
        step = graph.next_step_toward(current_node_id, goal_node, weather, None, use_weighted=False)
        if step and step in available:
            return step
        # Pick neighbor with lowest weighted cost to goal
        best_alt = None
        best_alt_cost = float('inf')
        for n in available:
            path = graph.weighted_shortest_path(n, goal_node, weather, soft_blocked, remaining_process_nodes)
            if path:
                cost = sum(graph.edge_cost(path[i], path[i+1], weather, soft_blocked, remaining_process_nodes)
                           for i in range(len(path)-1))
                if cost < best_alt_cost:
                    best_alt_cost = cost
                    best_alt = n
        if best_alt:
            return best_alt

    # No goal: fall back to first available neighbor
    return available[0]


def _handle_contesting(
    match_id: str, round_num: int, player_id: int,
    player: dict, contests: list[dict] | None,
    events: list[dict] | None, active_contest_id: str,
    my_player: dict, all_players: list[dict], phase: str,
    on_water_route: bool = False,
) -> dict:
    """Handle CONTESTING state: choose window card (策略文档 §7)."""
    contest_id = _find_contest_id(player_id, contests, events, active_contest_id)
    if not contest_id:
        return make_empty_action(match_id, round_num, player_id)

    # Determine contest type and pick card
    contest = _find_contest(contest_id, contests)
    contest_type = ""
    if contest:
        contest_type = contest.get("contestType") or contest.get("type", "")

    card = _choose_window_card(contest_type, contest, my_player, all_players, phase, on_water_route)
    return make_action(match_id, round_num, player_id, [
        make_window_card_action(contest_id, card)
    ])


def _find_contest_id(
    player_id: int,
    contests: list[dict] | None,
    events: list[dict] | None,
    active_contest_id: str,
) -> str:
    """Find the contest ID for the current player."""
    if active_contest_id:
        return active_contest_id
    if contests:
        for c in contests:
            if c.get("redPlayerId") == player_id or c.get("bluePlayerId") == player_id:
                if not c.get("resolved", False) and c.get("status") != "SUPPRESSED":
                    return c.get("contestId", "")
    if events:
        for ev in reversed(events):
            if ev.get("type") == "WINDOW_CONTEST_START":
                payload = ev.get("payload", {})
                cid = payload.get("contestId", "")
                if cid:
                    return cid
    return ""


def _find_contest(contest_id: str, contests: list[dict] | None) -> dict | None:
    """Find contest dict by ID."""
    if contests:
        for c in contests:
            if c.get("contestId") == contest_id:
                return c
    return None


# 窗口牌克制表 (任务书 §5.4.4): CARD_BEATS[c] = c 能击败的牌集合。
# 献贡/兵争各克 2 张、只怕 1 张(强牌); 验牒/强行各克 1 张、怕 2 张(弱牌)。
CARD_BEATS: dict[str, set[str]] = {
    "YAN_DIE": {"QIANG_XING"},
    "QIANG_XING": {"XIAN_GONG"},
    "XIAN_GONG": {"YAN_DIE", "BING_ZHENG"},
    "BING_ZHENG": {"YAN_DIE", "QIANG_XING"},
}

# 各牌的相对成本罚分(资源稀缺度)。价值高的窗口会弱化罚分。
# 兵争消耗护卫行动点(全局仅4点,最稀缺); 强行会消耗宝贵马匹(应留给加速);
# 献贡消耗1好果; 验牒消耗文书资源(过所/官凭,本就用于出牌,最廉价)。
CARD_COST_PENALTY: dict[str, float] = {
    "YAN_DIE": 1.0,
    "XIAN_GONG": 1.6,
    "QIANG_XING": 2.6,
    "BING_ZHENG": 3.2,
    "ABSTAIN": 0.0,
}

_ALL_EFFECTIVE_CARDS = ("YAN_DIE", "QIANG_XING", "XIAN_GONG", "BING_ZHENG")


def _window_card_result(mine: str, theirs: str) -> int:
    """本拍胜负 (任务书 §5.4.4): 胜=1, 平=0, 负=-1。"""
    if mine == theirs:
        return 0
    if theirs == "ABSTAIN":
        return 1 if mine != "ABSTAIN" else 0
    if mine == "ABSTAIN":
        return -1
    if theirs in CARD_BEATS.get(mine, ()):
        return 1
    if mine in CARD_BEATS.get(theirs, ()):
        return -1
    return 0


def _affordable_window_cards(p: dict | None) -> set[str]:
    """依据公开状态推断某方本拍买得起哪些有效牌 (任务书 §5.4.3)。"""
    if not p:
        return set()
    cards: set[str] = set()
    resources = get_player_resources(p)
    if resources.get("PASS_TOKEN", 0) + resources.get("OFFICIAL_PERMIT", 0) > 0:
        cards.add("YAN_DIE")
    if has_resource(p, "FAST_HORSE") or has_resource(p, "SHORT_HORSE"):
        cards.add("QIANG_XING")
    if get_good_fruit(p) >= 1 and get_freshness(p) >= 80:
        cards.add("XIAN_GONG")
    if get_action_points(p) > 0:
        cards.add("BING_ZHENG")
    return cards


def _contest_value_profile(
    contest_type: str, contest: dict | None, on_water_route: bool,
) -> tuple[bool, float]:
    """返回 (是否必争, 价值权重 0..1)。价值越高越值得付成本、越可能弃权亏。"""
    if contest_type == "GATE":
        return True, 1.0
    if contest_type == "PASS":
        return False, 0.8
    if contest_type in ("TASK", "OBSTACLE"):
        score = contest.get("taskScore", 0) if contest else 0
        return (False, 0.6) if score >= 30 else (False, 0.0)
    if contest_type == "DOCK":
        return (False, 0.5) if on_water_route else (False, 0.0)
    if contest_type == "RESOURCE":
        return False, 0.45
    return False, 0.0


def _choose_window_card(
    contest_type: str, contest: dict | None,
    my_player: dict, all_players: list[dict], phase: str,
    on_water_route: bool = False,
) -> str:
    """博弈式窗口出牌 (任务书 §5.4)。

    核心: 对手资源/好果/鲜度/行动点均为公开状态, 由此推断对手本拍能出哪些牌,
    再用正确克制表做期望胜点最大化, 同时按窗口价值权衡稀缺资源成本。
    不写死固定优先级, 随对手手牌与窗口价值自适应。
    """
    must_win, value = _contest_value_profile(contest_type, contest, on_water_route)

    my_cards = _affordable_window_cards(my_player)
    # 低价值且非必争的窗口: 直接弃权, 保留资源。
    if not must_win and value <= 0.0:
        return "ABSTAIN"
    if not my_cards:
        return "ABSTAIN"

    opp = _find_opponent(all_players, my_player.get("playerId"))
    opp_cards = _affordable_window_cards(opp)

    # 估计对手本拍出牌的概率权重。
    opp_weights: dict[str, float] = {}
    if opp is None:
        # 完全观测不到对手对象(信息缺失): 保守假设各有效牌均可能。
        for c in _ALL_EFFECTIVE_CARDS:
            opp_weights[c] = 0.5
        opp_weights["ABSTAIN"] = 1.0
    elif opp_cards:
        for c in opp_cards:
            opp_weights[c] = 1.0
        # 窗口越重要, 对手越可能认真出牌 → 弃权权重越低。
        opp_weights["ABSTAIN"] = max(0.1, 1.0 - value)
    else:
        # 公开状态显示对手买不起任何有效牌 → 对手本拍只能弃权,
        # 此时任意有效牌皆胜, 交由成本罚分挑最省的一张。
        opp_weights["ABSTAIN"] = 1.0

    total_w = sum(opp_weights.values()) or 1.0

    def expected(card: str) -> float:
        return sum(w * _window_card_result(card, o) for o, w in opp_weights.items()) / total_w

    best_card = "ABSTAIN"
    best_adj = float("-inf") if must_win else 0.0
    for card in my_cards:
        exp = expected(card)
        # 成本罚分随窗口价值弱化, 但始终保留一个下限, 使期望相等时优选省资源的牌。
        penalty = CARD_COST_PENALTY.get(card, 1.0) * max(0.03, (1.0 - value) * 0.2)
        adj = exp - penalty
        if adj > best_adj:
            best_adj = adj
            best_card = card

    # 必争窗口(如宫门)绝不弃权: 退化为纯期望最高的有效牌。
    if must_win and best_card == "ABSTAIN":
        best_card = max(my_cards, key=expected)

    return best_card


def _get_pending_station_process_type(
    current_node_id: str | None,
    next_node_id: str,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> str:
    if not current_node_id or next_node_id or not process_nodes:
        return ""
    if current_node_id in processed_node_ids:
        return ""

    process_type = process_nodes.get(current_node_id, {}).get("processType")
    if process_type and not is_verify_process(process_type):
        return process_type
    return ""


def _has_current_process_for_node(player: dict, current_node_id: str | None) -> bool:
    if not current_node_id:
        return False
    current_process = player.get("currentProcess")
    if not isinstance(current_process, dict):
        return False
    target_node_id = current_process.get("targetNodeId", "")
    object_key = current_process.get("objectKey", "")
    return target_node_id == current_node_id or object_key.startswith(f"PROCESS:{current_node_id}:")


def _resolve_guard_block_target(
    player: dict,
    route_blocked: set[str],
    guard_blocked_targets: set[str],
) -> str:
    """Node blocking our in-progress move (next hop or known guard)."""
    next_node = player.get("nextNodeId", "")
    if next_node and next_node in route_blocked:
        return next_node
    return ""


def _make_squad_weaken_action(
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
    player_id: int,
    player: dict,
) -> dict | None:
    if not target_node_id or get_squad_count(player) < 2:
        return None
    for node in inquire_nodes:
        if node.get("nodeId") != target_node_id:
            continue
        guard = node.get("guard", {})
        if is_enemy_guard(guard, my_team_id, player_id):
            if guard.get("defense", 0) > 0:
                return make_squad_weaken_action(target_node_id)
            return None
    # inquire 可能未包含远程节点，仍尝试削弱
    return make_squad_weaken_action(target_node_id)


def _wait_and_weaken_guard(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
) -> dict:
    """WAIT (主车队) + SQUAD_WEAKEN (小分队) 每帧削弱设卡直到通行。"""
    msg = make_action(match_id, round_num, player_id, [make_wait_action()])
    squad = _make_squad_weaken_action(
        inquire_nodes, target_node_id, my_team_id, player_id, player,
    )
    if squad:
        logger.info("Round %d: WAIT + squad weaken at %s", round_num, target_node_id)
        return _append_squad_action(msg, squad)
    return msg


def _move_and_weaken_guard(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
) -> dict:
    """Resume a paused edge move while optionally weakening the blocking guard."""
    msg = make_action(match_id, round_num, player_id, [make_move_action(target_node_id)])
    squad = _make_squad_weaken_action(
        inquire_nodes, target_node_id, my_team_id, player_id, player,
    )
    if squad:
        logger.info("Round %d: MOVE + squad weaken at %s", round_num, target_node_id)
        return _append_squad_action(msg, squad)
    logger.info("Round %d: MOVE to %s while checking route block", round_num, target_node_id)
    return msg


def _continue_and_weaken_guard(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
) -> dict:
    """Keep natural movement progress while optionally weakening the guard."""
    msg = make_empty_action(match_id, round_num, player_id)
    squad = _make_squad_weaken_action(
        inquire_nodes, target_node_id, my_team_id, player_id, player,
    )
    if squad:
        logger.info("Round %d: continuing move + squad weaken at %s", round_num, target_node_id)
        return _append_squad_action(msg, squad)
    logger.info("Round %d: continuing move while checking route block at %s", round_num, target_node_id)
    return msg


def _handle_moving(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, weather: dict | None, phase: str,
) -> dict:
    """Handle MOVING state: can use horse or rush_speed."""
    # Use FAST_HORSE if available and on a long road segment
    if has_resource(player, "FAST_HORSE"):
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("FAST_HORSE")
        ])
    # Use SHORT_HORSE if available
    if has_resource(player, "SHORT_HORSE"):
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("SHORT_HORSE")
        ])
    return make_empty_action(match_id, round_num, player_id)


def _handle_blocked_by_guard(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None, inquire_nodes: list[dict],
    process_nodes: dict[str, dict] | None = None,
) -> dict:
    """Handle MOVE_BLOCKED_BY_GUARD error (策略文档 §3.4)."""
    if blocked is None:
        blocked = set()
    neighbors = graph.get_neighbors(current_node_id)
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")

    # Detour via unblocked neighbor (e.g. S09→S05 when S10 guarded)
    best_detour = None
    best_cost = float("inf")
    for n in neighbors:
        if n in blocked:
            continue
        if not goal:
            return make_action(match_id, round_num, player_id, [make_move_action(n)])
        path = graph.weighted_shortest_path(n, goal, weather, blocked, process_nodes)
        if path:
            cost = sum(
                graph.edge_cost(path[i], path[i + 1], weather, blocked, process_nodes)
                for i in range(len(path) - 1)
            )
            if cost < best_cost:
                best_cost = cost
                best_detour = n
    if best_detour:
        logger.info("Round %d: Detour via %s to avoid guard", round_num, best_detour)
        return make_action(match_id, round_num, player_id, [make_move_action(best_detour)])

    # No detour: BREAK_GUARD or FORCED_PASS on guarded neighbor
    for n in neighbors:
        if n not in blocked:
            continue
        for node in inquire_nodes:
            if node.get("nodeId") != n:
                continue
            guard = node.get("guard", {})
            if is_enemy_guard(guard, get_team_id(player), player_id):
                good = min(get_good_fruit(player), 2)
                bad = min(get_bad_fruit(player), 2)
                if good + bad > 0:
                    logger.info(
                        "Round %d: BREAK_GUARD at %s (gf=%d bf=%d)",
                        round_num, n, good, bad,
                    )
                    return make_action(match_id, round_num, player_id, [
                        make_break_guard_action(n, good_fruit=good, bad_fruit=bad)
                    ])
        logger.info("Round %d: FORCED_PASS at guarded %s", round_num, n)
        return make_action(match_id, round_num, player_id, [
            make_forced_pass_action(n)
        ])

    return make_empty_action(match_id, round_num, player_id)


def _retry_task_at_current_node(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    tasks: list[dict],
    failed_task_ids: set[str],
    preferred_task_id: str = "",
) -> dict | None:
    if get_task_score(player) >= TASK_SCORE_TARGET:
        return None
    if isinstance(player.get("currentProcess"), dict):
        return None

    neighbors = graph.get_neighbors(current_node_id) if graph else None
    task = None
    if preferred_task_id:
        for candidate in tasks:
            if candidate.get("taskId", "") == preferred_task_id:
                task_node = candidate.get("nodeId", "")
                if (
                    task_node == current_node_id
                    or (neighbors is not None and task_node in neighbors and get_task_template_id(candidate).startswith("T04"))
                ):
                    task = candidate
                break
    if not task:
        task = find_task_at_node(
            tasks, current_node_id, player_id,
            graph_neighbors=neighbors,
        )
    if not task:
        return None
    if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
        return None
    owner = task.get("ownerPlayerId", 0)
    if owner != 0 and owner != player_id:
        return None
    protection = task.get("protectionPlayerId", 0)
    if protection != 0 and protection != player_id:
        return None

    task_id = task.get("taskId", "")
    if not task_id or task_id in failed_task_ids:
        return None
    template_id = get_task_template_id(task)
    if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
        return None
    expire_round = task.get("expireRound", 0)
    if expire_round > 0 and round_num >= expire_round:
        return None

    logger.info("Round %d: Retrying task %s (template=%s) at %s", round_num, task_id, template_id, current_node_id)
    return make_action(match_id, round_num, player_id, [make_claim_task_action(task_id)])


def _handle_tasks(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    tasks: list[dict], my_player_id: int, phase: str,
    weather: dict | None, blocked: set[str] | None,
    goal_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    failed_task_ids: set[str] | None = None,
    global_plan: GlobalPlan | None = None,
) -> dict | None:
    """Handle task claiming strategy (策略文档 §5).

    Returns action dict or None.
    """
    if terminal_node_ids is None:
        terminal_node_ids = []
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if failed_task_ids is None:
        failed_task_ids = set()

    my_task_score = get_task_score(player)
    if _should_force_delivery(round_num, phase, player, global_plan):
        return None

    # Already at stretch target, don't need more tasks
    if my_task_score >= TASK_SCORE_STRETCH and phase != "RUSH":
        return None

    # Check if we're currently processing a task (策略文档 §5.2: 同时仅处理1个任务实例)
    for task in tasks:
        if (task.get("ownerPlayerId") == my_player_id
                and task.get("active", False)
                and not task.get("completed", False)
                and not task.get("failed", False)):
            return None

    # Try to claim task at current node (prioritized by score/round)
    task = find_task_at_node(
        tasks, current_node_id, my_player_id,
        graph_neighbors=graph.get_neighbors(current_node_id) if graph else None,
    )
    if task:
        template_id = get_task_template_id(task)
        if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
            logger.debug("Round %d: Skipping T06 task (no horse)", round_num)
            task = None

    if task:
        # Check expireRound (策略文档 §5.2: 关注expireRound)
        expire_round = task.get("expireRound", 0)
        if expire_round > 0 and round_num >= expire_round:
            logger.debug("Round %d: Task %s expired", round_num, task.get("taskId", ""))
            task = None

    if task:
        # Skip tasks previously rejected with RESOURCE_NOT_ENOUGH
        task_id = task.get("taskId", "")
        if task_id and task_id in failed_task_ids:
            logger.debug("Round %d: Skipping failed task %s", round_num, task_id)
            task = None

    if task:
        task_id = task.get("taskId", "")
        if task_id:
            net_value = task_net_value(task, 0.0, round_num, global_plan, is_current_node=True) if global_plan else 999.0
            if global_plan and net_value < MIN_TASK_NET_VALUE:
                logger.info(
                    "Round %d: Skipping local task %s net=%.1f plan=%s",
                    round_num, task_id, net_value, global_plan.reason,
                )
                return None
            logger.info(
                "Round %d: Claiming task %s (template=%s) at %s net=%.1f",
                round_num, task_id, template_id, current_node_id, net_value,
            )
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(task_id)
            ])

    # Look for nearby tasks within detour cost (策略文档 §5.2 顺路原则)
    if my_task_score < TASK_SCORE_TARGET:
        candidates = []
        for task in tasks:
            if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
                continue
            owner = task.get("ownerPlayerId", 0)
            if owner != 0 and owner != my_player_id:
                continue
            protection = task.get("protectionPlayerId", 0)
            if protection != 0 and protection != my_player_id:
                continue
            task_node = task.get("nodeId", "")
            if not task_node:
                continue
            if task_node in visited_node_ids:
                tid = get_task_template_id(task)
                if not (tid.startswith("T04") and task_node in obstacle_nodes):
                    continue

            # T06: skip if no horse
            tid = get_task_template_id(task)
            if tid.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
                continue

            # Skip tasks previously rejected with RESOURCE_NOT_ENOUGH
            if task.get("taskId", "") in failed_task_ids:
                continue

            # Check expireRound
            expire_round = task.get("expireRound", 0)
            if expire_round > 0 and round_num >= expire_round:
                continue

            # Check detour cost
            detour = _calc_detour_cost(graph, current_node_id, task_node, goal_node_id, terminal_node_ids, weather, blocked, player, process_nodes)
            if detour <= MAX_TASK_DETOUR_COST:
                # Score per round priority (策略文档 §5.1)
                spr = 0.0
                for prefix, (score, proc_round, score_per_round) in TASK_PRIORITY.items():
                    if tid.startswith(prefix):
                        spr = score_per_round
                        break
                net_value = (
                    task_net_value(task, detour, round_num, global_plan, is_current_node=False)
                    if global_plan else spr - detour
                )
                if net_value >= MIN_TASK_NET_VALUE:
                    candidates.append((task, detour, net_value))

        if candidates:
            # Sort by global net value descending, then detour ascending
            candidates.sort(key=lambda x: (-x[2], x[1]))
            best_task = candidates[0][0]
            task_node = best_task.get("nodeId", "")
            # Move toward the task node using weighted routing, avoid backtracking
            soft_blocked = set(obstacle_nodes)
            soft_blocked.update(visited_node_ids)
            soft_blocked.discard(task_node)  # Don't block the target
            step = graph.next_step_toward(current_node_id, task_node, weather, soft_blocked, use_weighted=True, process_nodes=process_nodes)
            if not step:
                # Fallback without soft-blocked
                step = graph.next_step_toward(current_node_id, task_node, weather, obstacle_nodes, use_weighted=True, process_nodes=process_nodes)
            if step:
                logger.info(
                    "Round %d: Moving toward task at %s (template=%s), step=%s net=%.1f",
                    round_num, task_node, get_task_template_id(best_task), step, candidates[0][2],
                )
                return make_action(match_id, round_num, player_id, [make_move_action(step)])

    return None


def _calc_detour_cost(
    graph: MapGraph, current: str, task_node: str,
    gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    player: dict,
    process_nodes: dict[str, dict] | None = None,
) -> int:
    """Calculate the extra weighted cost of detouring to a task node vs direct route."""
    goal = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current, weather, blocked, process_nodes)
    if not goal:
        return 999

    def _weighted_path_cost(a: str, b: str) -> float:
        path = graph.weighted_shortest_path(a, b, weather, blocked, process_nodes)
        if not path:
            return float('inf')
        return sum(graph.edge_cost(path[i], path[i+1], weather, blocked, process_nodes)
                   for i in range(len(path)-1))

    direct = _weighted_path_cost(current, goal)
    via_task = _weighted_path_cost(current, task_node) + _weighted_path_cost(task_node, goal)

    if direct == float('inf') or via_task == float('inf'):
        return 999

    return max(0, int(via_task - direct))


def _handle_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    current_node: dict | None, phase: str, weather: dict | None,
    global_plan: GlobalPlan | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    inquire_nodes: list[dict] | None = None,
    map_profile: MapProfile | None = None,
) -> dict | None:
    """Handle resource claiming strategy (策略文档 §6).

    Returns action dict or None.
    """
    if current_node is None:
        return None
    if phase == "RUSH" or round_num >= 360:
        return None

    resources = find_available_resources(current_node)
    if not resources:
        return None

    my_resources = get_player_resources(player)
    route = None
    if global_plan and gate_node_id:
        route = estimate_delivery_route(
            graph, current_node_id, player, gate_node_id, terminal_node_ids or [],
            weather, process_nodes, processed_node_ids or set(), None,
        )

    # Filter to only high-value resources worth claiming
    HIGH_VALUE_RESOURCES = {"FAST_HORSE", "SHORT_HORSE", "ICE_BOX"}
    WINDOW_RESOURCES = {"OFFICIAL_PERMIT", "PASS_TOKEN"}
    # GUARD_RESERVE_FOR_GATE: reserve 1 permit for S14 GATE contest (策略文档 §15)
    PERMIT_RESERVE = 1

    for rtype, count in resources:
        # Skip if already have this resource
        if my_resources.get(rtype, 0) >= 1 and rtype in HIGH_VALUE_RESOURCES:
            continue
        # Only claim high-value resources (FAST_HORSE, SHORT_HORSE, ICE_BOX)
        if rtype in HIGH_VALUE_RESOURCES:
            value = resource_net_value(rtype, player, global_plan, route, map_profile) if global_plan else 999.0
            if value >= MIN_RESOURCE_NET_VALUE:
                logger.info(
                    "Round %d: Claiming resource %s at %s value=%.1f",
                    round_num, rtype, current_node_id, value,
                )
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
            continue
        # Claim OFFICIAL_PERMIT/PASS_TOKEN for window contests
        # Keep at least PERMIT_RESERVE+1 (1 for current use + reserve for GATE)
        if rtype in WINDOW_RESOURCES:
            total_permits = my_resources.get("OFFICIAL_PERMIT", 0) + my_resources.get("PASS_TOKEN", 0)
            value = resource_net_value(rtype, player, global_plan, route, map_profile) if global_plan else 999.0
            if total_permits < PERMIT_RESERVE + 1 and value >= MIN_RESOURCE_NET_VALUE:
                logger.info(
                    "Round %d: Claiming resource %s at %s (for window contests) value=%.1f",
                    round_num, rtype, current_node_id, value,
                )
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
            continue
        # Claim BOAT_RIGHT (策略文档 §6.1: 仅领取, passive)
        if rtype == "BOAT_RIGHT" and my_resources.get("BOAT_RIGHT", 0) < 1:
            value = resource_net_value(rtype, player, global_plan, route, map_profile) if global_plan else 999.0
            if value >= MIN_RESOURCE_NET_VALUE:
                logger.info("Round %d: Claiming BOAT_RIGHT at %s value=%.1f", round_num, current_node_id, value)
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
        # Skip INTEL — low value, not worth the frames early game

    if global_plan and gate_node_id and inquire_nodes:
        detour_candidates: list[tuple[float, str, str, float]] = []
        def _route_cost(start: str, end: str) -> float:
            path = graph.weighted_shortest_path(start, end, weather, None, process_nodes)
            if not path:
                return float("inf")
            return sum(
                graph.edge_cost(path[i], path[i + 1], weather, None, process_nodes)
                for i in range(len(path) - 1)
            )

        for node in inquire_nodes:
            node_id = node.get("nodeId", "")
            if not node_id or node_id == current_node_id:
                continue
            for rtype, _count in find_available_resources(node):
                cost_to_resource = _route_cost(current_node_id, node_id)
                route_after_resource = estimate_delivery_route(
                    graph, node_id, player, gate_node_id, terminal_node_ids or [],
                    weather, process_nodes, processed_node_ids or set(), None,
                )
                if cost_to_resource == float("inf") or route_after_resource.cost == float("inf"):
                    continue
                direct = global_plan.direct_eta
                detour = max(0.0, cost_to_resource + route_after_resource.cost - direct)
                if detour > 24:
                    continue
                value = resource_net_value(rtype, player, global_plan, route_after_resource, map_profile) - detour * 0.65
                if value >= MIN_RESOURCE_NET_VALUE:
                    detour_candidates.append((value, node_id, rtype, detour))
        if detour_candidates:
            detour_candidates.sort(key=lambda item: (-item[0], item[3]))
            value, node_id, rtype, detour = detour_candidates[0]
            step = graph.next_step_toward(
                current_node_id, node_id, weather, None,
                use_weighted=True, process_nodes=process_nodes,
            )
            if step:
                logger.info(
                    "Round %d: Moving toward resource %s at %s step=%s value=%.1f detour=%.1f",
                    round_num, rtype, node_id, step, value, detour,
                )
                return make_action(match_id, round_num, player_id, [make_move_action(step)])

    return None


def _handle_force_delivery_resource(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    current_node: dict | None,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    global_plan: GlobalPlan | None = None,
    map_profile: MapProfile | None = None,
) -> dict | None:
    """Claim only resources that directly shorten the forced delivery route."""
    if current_node is None or has_resource(player, "FAST_HORSE"):
        return None
    direct_target = _find_direct_delivery_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids,
    )
    if not direct_target:
        return None
    edge = graph.get_edge(current_node_id, direct_target)
    edge_distance = edge.get("distance", 0) if edge else 0
    route_type = graph.get_edge_route_type(current_node_id, direct_target)
    route = estimate_delivery_route(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids, None,
    )
    for rtype, _count in find_available_resources(current_node):
        value = resource_net_value(rtype, player, global_plan, route, map_profile) if global_plan else 999.0
        if (
            rtype == "FAST_HORSE"
            and value >= MIN_RESOURCE_NET_VALUE
            and (edge_distance >= 30 or route_type in {"ROAD", "MOUNTAIN", "BRANCH"})
        ):
            logger.info(
                "Round %d: FORCE_DELIVERY claiming FAST_HORSE at %s for %s->%s value=%.1f",
                round_num, current_node_id, current_node_id, direct_target, value,
            )
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
    return None


def _handle_use_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node_id: str, graph: MapGraph,
    weather: dict | None, phase: str,
    global_plan: GlobalPlan | None = None,
) -> dict | None:
    """Handle using resources: ice box, horses (策略文档 §6.1)."""
    freshness = get_freshness(player)
    force_delivery = _should_force_delivery(round_num, phase, player, global_plan)

    # Use ICE_BOX when freshness is low or preemptively before bad weather/routes
    # (策略文档 §6.1: 鲜度<72 或酷暑/山路前)
    if has_resource(player, "ICE_BOX"):
        use_ice = False
        if force_delivery and freshness >= 20:
            use_ice = False
        elif freshness < ICE_BOX_FRESHNESS_THRESHOLD:
            use_ice = True
        # Preemptive: check if next route segment is mountain or hot weather
        elif weather and freshness < 80:
            forecasts = weather.get("forecast", [])
            for fw in forecasts:
                wtype = fw.get("type", "")
                if wtype == "HOT":
                    use_ice = True
                    break
        if not use_ice:
            # Check if next step goes through mountain
            neighbors = graph.get_neighbors(current_node_id)
            for n in neighbors:
                if graph.get_edge_route_type(current_node_id, n) == "MOUNTAIN" and freshness < 80:
                    use_ice = True
                    break
        if use_ice:
            logger.info("Round %d: Using ICE_BOX (freshness=%.1f)", round_num, freshness)
            return make_action(match_id, round_num, player_id, [
                make_use_resource_action("ICE_BOX")
            ])

    # Save horse buffs for forced delivery; using them mid-route wastes the short duration.
    if force_delivery and has_resource(player, "FAST_HORSE"):
        neighbors = graph.get_neighbors(current_node_id)
        for n in neighbors:
            edge = graph.get_edge(current_node_id, n)
            distance = edge.get("distance", 0) if edge else 0
            if graph.get_edge_route_type(current_node_id, n) in {"ROAD", "MOUNTAIN", "BRANCH"} and distance >= 25:
                logger.info("Round %d: Using FAST_HORSE before long move %s->%s", round_num, current_node_id, n)
                return make_action(match_id, round_num, player_id, [
                    make_use_resource_action("FAST_HORSE")
                ])

    if force_delivery and has_resource(player, "SHORT_HORSE") and not has_resource(player, "FAST_HORSE"):
        neighbors = graph.get_neighbors(current_node_id)
        if neighbors:
            logger.info("Round %d: Using SHORT_HORSE before move", round_num)
            return make_action(match_id, round_num, player_id, [
                make_use_resource_action("SHORT_HORSE")
            ])

    return None


def _handle_combat(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    mode: str, phase: str, inquire_nodes: list[dict],
    opp_player: dict | None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    visited_node_ids: set[str] | None = None,
    my_team_id: str = "",
    global_plan: GlobalPlan | None = None,
) -> dict | None:
    """Handle combat: guard, break, squad (策略文档 §8)."""
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if process_nodes is None:
        process_nodes = {}
    if visited_node_ids is None:
        visited_node_ids = set()
    if not my_team_id:
        my_team_id = get_team_id(player)

    # SET_GUARD 仅在关口争夺或 RUSH 阶段守宫门时（开局设卡浪费帧数）
    global_guard = False
    if global_plan is not None:
        global_guard = should_set_guard_now(
            player, opp_player, current_node_id, graph,
            gate_node_id, terminal_node_ids, weather, blocked,
            inquire_nodes, global_plan,
        )
    should_set_guard = (
        mode == "GATE_FIGHT"
        or (phase == "RUSH" and gate_node_id and current_node_id == gate_node_id)
        or global_guard
    )
    if should_set_guard and get_good_fruit(player) >= 1:
        guard_target = _find_guard_target(
            graph, current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked, player, inquire_nodes, my_team_id,
        )
        if guard_target and guard_target == current_node_id:
            logger.info(
                "Round %d: Setting guard at current node %s global=%s",
                round_num, guard_target, global_guard,
            )
            extra = 1 if get_good_fruit(player) >= 2 else 0
            return make_action(match_id, round_num, player_id, [
                make_set_guard_action(guard_target, extra_good_fruit=extra)
            ])

    # --- BREAK_GUARD with optional BREAK_ORDER (策略文档 §8.2, §10) ---
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if goal and blocked:
        optimal_path = graph.shortest_path(current_node_id, goal, weather, obstacle_nodes)
        if optimal_path and len(optimal_path) >= 2:
            next_hop = optimal_path[1]
            if next_hop in blocked:
                for node in inquire_nodes:
                    if node.get("nodeId") == next_hop:
                        guard = node.get("guard", {})
                        if is_enemy_guard(guard, my_team_id, player_id):
                            good = min(get_good_fruit(player), 2)
                            bad = min(get_bad_fruit(player), 2)
                            if good + bad > 0:
                                action = make_break_guard_action(next_hop, good_fruit=good, bad_fruit=bad)
                                # Bind BREAK_ORDER if in RUSH and have resources (策略文档 §10: +3攻坚)
                                if phase == "RUSH" and (bad >= 2 or good >= 1):
                                    action["rushTactic"] = "BREAK_ORDER"
                                    logger.info("Round %d: Breaking guard at %s with BREAK_ORDER", round_num, next_hop)
                                else:
                                    logger.info("Round %d: Breaking guard at %s (blocking path)", round_num, next_hop)
                                return make_action(match_id, round_num, player_id, [action])
                        # Try FORCED_PASS instead
                        logger.info("Round %d: Forced pass at %s", round_num, next_hop)
                        return make_action(match_id, round_num, player_id, [
                            make_forced_pass_action(next_hop)
                        ])

    # --- Squad actions (策略文档 §8.4) — only if not RUSH ---
    if phase != "RUSH":
        squad_count = get_squad_count(player)
        my_task_score = get_task_score(player)

        # Reserve squads for late key-pass guard weakening; scouting is optional.
        if squad_count >= 10 and my_task_score < TASK_SCORE_TARGET and process_nodes:
            for nid, info in process_nodes.items():
                if nid not in visited_node_ids and nid != current_node_id:
                    dist = graph.path_length(current_node_id, nid, weather, None)
                    if 0 < dist <= 15:
                        logger.info("Round %d: Squad scout at %s", round_num, nid)
                        return make_action(match_id, round_num, player_id, [
                            make_squad_scout_action(nid)
                        ])

        # SQUAD_CLEAR: Clear obstacles without main team (策略文档 §8.4: 2人手)
        if squad_count >= 8:
            for node in inquire_nodes:
                if node.get("hasObstacle", False) and node.get("nodeId") != current_node_id:
                    nid = node.get("nodeId", "")
                    # Check if obstacle is on our path
                    if goal:
                        path = graph.shortest_path(current_node_id, goal, weather, obstacle_nodes)
                        if path and nid in path:
                            logger.info("Round %d: Squad clear at %s", round_num, nid)
                            return make_action(match_id, round_num, player_id, [
                                make_squad_clear_action(nid)
                            ])

        # SQUAD_REINFORCE: Reinforce our own guard at key nodes (策略文档 §8.4: 2人手)
        if squad_count >= 8:
            for node in inquire_nodes:
                guard = node.get("guard", {})
                owner_team = guard.get("ownerTeamId") if guard else ""
                if (guard and owner_team == my_team_id
                        and guard_is_active(guard)
                        and node.get("nodeId") != current_node_id):
                    nid = node.get("nodeId", "")
                    logger.info("Round %d: Squad reinforce at %s", round_num, nid)
                    return make_action(match_id, round_num, player_id, [
                        make_squad_reinforce_action(nid)
                    ])

        # SQUAD_WEAKEN: Weaken enemy guard (策略文档 §8.4: 2人手, 性价比高)
        if squad_count >= 2 and opp_player:
            for node in inquire_nodes:
                guard = node.get("guard", {})
                if (is_enemy_guard(guard, my_team_id, player_id)
                        and node.get("nodeId") != current_node_id):
                    nid = node.get("nodeId", "")
                    logger.info("Round %d: Squad weaken at %s", round_num, nid)
                    return make_action(match_id, round_num, player_id, [
                        make_squad_weaken_action(nid)
                    ])

    return None


def _find_guard_target(
    graph: MapGraph, current_node_id: str,
    gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    player: dict, inquire_nodes: list[dict],
    my_team_id: str,
) -> str | None:
    """Find a good node to set guard on (策略文档 §8.1).

    Key: don't set guard on our own route. Target opponent's likely route.
    """
    # Find nodes that are NOT on our path to gate/terminal
    goal = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current_node_id, weather, blocked)
    if not goal:
        return None

    our_path = graph.weighted_shortest_path(current_node_id, goal, weather, blocked)
    if our_path and current_node_id in our_path[-3:]:
        return None

    neighbors = graph.get_neighbors(current_node_id)
    if len(neighbors) <= 3:
        for node in inquire_nodes:
            if node.get("nodeId") == current_node_id and not guard_is_active(node.get("guard")):
                return current_node_id

    return None


def _handle_rush_tactics(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node_id: str, phase: str, mode: str,
    graph: MapGraph | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    weather: dict | None = None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
) -> dict | None:
    """Handle rush tactics: RUSH_SPEED, RUSH_PROTECT (策略文档 §10).

    Only available after RUSH phase. Each can be used once per match.
    RUSH_SPEED与马互斥: 有马buff时不使用.
    """
    if phase != "RUSH":
        return None

    # RUSH_SPEED can only be used when IDLE
    state = player.get("state", "")
    if state != "IDLE":
        return None

    freshness = get_freshness(player)

    # RUSH_PROTECT: 鲜度<50, 停靠节点使用 (策略文档 §10: 0成本)
    if freshness < RUSH_PROTECT_FRESHNESS:
        logger.info("Round %d: Using RUSH_PROTECT (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [make_rush_protect_action()])

    # The current server rejects standalone RUSH_SPEED as INVALID_ACTION_TYPE.
    return None
