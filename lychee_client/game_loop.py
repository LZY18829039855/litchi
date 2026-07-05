"""Game loop: tying transport, messages, map, state, and strategy together."""

from __future__ import annotations

import logging
import socket
import time
from typing import Any

from lychee_client.transport import encode_frame, read_frames_from_buffer
from lychee_client.messages import parse_message, StartMessage, InquireMessage, OverMessage
from lychee_client.map_graph import MapGraph
from lychee_client.map_gameplay import MapGameplayContext, build_map_gameplay, default_map_gameplay
from lychee_client.state import can_move, get_current_node_id, needs_processing, GUARD_STUCK_AVOID_ROUNDS, get_enemy_busy_task_ids
from lychee_client.decision import make_registration, make_ready, make_action, make_empty_action
from lychee_client.strategy import decide_action

logger = logging.getLogger("lychee_client")


class GameClient:
    """Main game client that connects to the server and runs the game loop."""

    def __init__(self, host: str, port: int, player_id: int, player_name: str):
        self.host = host
        self.port = port
        self.player_id = player_id
        self.player_name = player_name
        self.sock: socket.socket | None = None
        self.recv_buffer = b""
        self.match_id = ""
        self.graph: MapGraph | None = None
        self.start_msg: StartMessage | None = None
        self.map_gameplay: MapGameplayContext = default_map_gameplay()
        self.process_nodes: dict[str, dict] = {}  # nodeId -> {processType, processRound}
        self.active_contest_id: str = ""  # cached contestId from WINDOW_CONTEST_START
        self.round_count = 0
        self.move_count = 0
        self.process_count = 0
        self.last_move_failed = False
        self.last_move_error = ""
        self.processed_node_ids: set[str] = set()  # nodes where we completed processing THIS visit (reset on leave)
        self.visited_node_ids: set[str] = set()  # all nodes ever visited (for navigation, avoid backtracking)
        self.failed_task_ids: set[str] = set()  # tasks rejected with RESOURCE_NOT_ENOUGH (skip retry)
        self.rush_speed_failed = False  # RUSH_SPEED rejected with INVALID_ACTION_TYPE (skip retry)
        self.last_claimed_task_id = ""  # track last CLAIM_TASK taskId for failed_task_ids
        self.last_claimed_task_node_id = ""
        self.pending_task_hold_task_id = ""
        self.pending_task_hold_node_id = ""
        self.pending_task_hold_until_round = 0
        self.guard_blocked_targets: set[str] = set()  # nodes blocked by enemy guard (for routing)
        self.avoid_route_nodes: set[str] = set()  # permanently avoided nodes after long guard stuck
        self.forced_pass_failed_targets: set[str] = set()  # targets where blind forced pass was rejected this stop
        self.squad_clear_pending: set[str] = set()  # obstacles already dispatched SQUAD_CLEAR
        self.own_guard_sites: set[str] = set()  # nodes where we placed SET_GUARD (advance separately)
        self.last_forced_pass_target = ""
        self.guard_stuck_target: str = ""
        self.guard_stuck_rounds: int = 0
        self.last_node_id: str = ""
        self.task_claimed_this_stop: bool = False
        self.start_round: int = 1
        self.running = False

    def connect(self) -> None:
        """Connect to the server via TCP."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect((self.host, self.port))
        logger.info("Connected to %s:%d", self.host, self.port)

    def send_message(self, msg: dict) -> None:
        """Send a message to the server with 5-digit length prefix."""
        frame = encode_frame(msg)
        if self.sock:
            self.sock.sendall(frame)
        logger.debug("Sent: %s", msg.get("msg_name", "?"))

    def receive_messages(self) -> list[dict]:
        """Receive and parse messages from the server, handling half/sticky packets."""
        messages = []
        # Try to read available data
        try:
            if self.sock:
                data = self.sock.recv(65536)
                if not data:
                    logger.info("Server closed connection")
                    self.running = False
                    return []
                self.recv_buffer += data
        except socket.timeout:
            return []

        # Parse complete frames from buffer
        parsed, self.recv_buffer = read_frames_from_buffer(self.recv_buffer)
        messages.extend(parsed)

        # If we parsed some but might have more, try one more read
        if parsed:
            try:
                if self.sock:
                    self.sock.settimeout(0.1)
                    data = self.sock.recv(65536)
                    if data:
                        self.recv_buffer += data
                        more, self.recv_buffer = read_frames_from_buffer(self.recv_buffer)
                        messages.extend(more)
                    self.sock.settimeout(30)
            except socket.timeout:
                self.sock.settimeout(30)

        return messages

    def send_registration(self) -> None:
        """Send registration message."""
        msg = make_registration(self.player_id, self.player_name)
        self.send_message(msg)
        logger.info("Sent registration as player %d", self.player_id)

    def handle_start(self, start: StartMessage) -> None:
        """Handle start message: cache match info and build map graph."""
        self.match_id = start.match_id
        self.start_round = start.round
        self.start_msg = start
        self.graph = MapGraph(start.nodes, start.edges)

        # Build process_nodes map from start message
        # Source 1: start.nodes[] with processType
        for node in start.nodes:
            nid = node.get("nodeId", "")
            pt = node.get("processType")
            if nid and pt:
                self.process_nodes[nid] = {
                    "processType": pt,
                    "processRound": node.get("processRound", 0),
                }

        # Source 2: start.map.gameplay.processNodes
        map_data = start.raw.get("map", {})
        gameplay = map_data.get("gameplay", {})
        for pn in gameplay.get("processNodes", []):
            nid = pn.get("nodeId", "")
            if nid and nid not in self.process_nodes:
                self.process_nodes[nid] = {
                    "processType": pn.get("processType", ""),
                    "processRound": pn.get("processRound", 0),
                }

        self.map_gameplay = build_map_gameplay(start.raw, self.graph)
        ctx = self.map_gameplay
        logger.info(
            "Received start: matchId=%s, %d nodes, %d edges, %d process nodes, "
            "map_kind=%s map_id=%s water=%s official_mid=%s obstacles=%s "
            "profile={ice=%.0f water_min=%d scout_min=%d force_buf=%.0f guard_min=%d}",
            start.match_id, len(start.nodes), len(start.edges), len(self.process_nodes),
            ctx.map_kind.value, ctx.map_id or "?",
            sorted(ctx.water_route_nodes),
            sorted(ctx.official_mid_route_nodes),
            sorted(ctx.obstacle_candidate_node_ids),
            ctx.profile.ice_box_freshness_threshold,
            ctx.profile.water_route_task_min,
            ctx.profile.squad_scout_min_squad,
            ctx.profile.force_delivery_slack_buffer,
            ctx.profile.guard_min_task_score,
        )

    def send_ready(self) -> None:
        """Send ready message."""
        msg = make_ready(self.match_id, self.start_round, self.player_id)
        self.send_message(msg)
        logger.info("Sent ready (round=%d)", self.start_round)

    def _sync_active_contest_id(self, inquire: InquireMessage) -> None:
        """Prefer live contests[] over cached event contestId."""
        for contest in inquire.contests:
            if contest.get("resolved", False):
                continue
            if contest.get("status") == "SUPPRESSED":
                continue
            red_id = contest.get("redPlayerId")
            blue_id = contest.get("bluePlayerId")
            if self.player_id not in (red_id, blue_id):
                continue
            cid = contest.get("contestId", "")
            if cid:
                self.active_contest_id = cid
                return
        if self.active_contest_id:
            still_active = any(
                c.get("contestId") == self.active_contest_id
                and not c.get("resolved", False)
                and c.get("status") != "SUPPRESSED"
                for c in inquire.contests
            )
            if not still_active:
                self.active_contest_id = ""

    def handle_inquire(self, inquire: InquireMessage) -> dict | None:
        """Handle inquire message: decide and send action.

        Returns the action message sent, or None if no action was sent.
        """
        self.round_count = inquire.round
        self._sync_active_contest_id(inquire)
        player = inquire.find_self_player(self.player_id)
        if player is None:
            logger.warning("Self player %d not found in inquire", self.player_id)
            return None

        current_node_id = player.get("currentNodeId")
        current_node = inquire.find_node(current_node_id) if current_node_id else None
        enemy_busy_task_ids = get_enemy_busy_task_ids(inquire.players, self.player_id)
        if (
            self.pending_task_hold_task_id
            and self.pending_task_hold_task_id in enemy_busy_task_ids
        ):
            logger.info(
                "Round %d: Clearing task hold for %s (enemy processing)",
                inquire.round, self.pending_task_hold_task_id,
            )
            self.pending_task_hold_task_id = ""
            self.pending_task_hold_node_id = ""
            self.pending_task_hold_until_round = 0

        # Track node changes
        # When leaving a node, remove it from processed_node_ids (§4.1: revisit requires re-process)
        # but keep it in visited_node_ids for navigation (avoid backtracking)
        if current_node_id and current_node_id != self.last_node_id:
            if self.last_node_id and self.last_node_id in self.processed_node_ids:
                self.processed_node_ids.discard(self.last_node_id)
            if self.pending_task_hold_node_id and self.pending_task_hold_node_id != current_node_id:
                self.pending_task_hold_task_id = ""
                self.pending_task_hold_node_id = ""
                self.pending_task_hold_until_round = 0
            self.forced_pass_failed_targets.clear()
            self.last_forced_pass_target = ""
            self.guard_blocked_targets.discard(current_node_id)
            self.avoid_route_nodes.discard(current_node_id)
            self.task_claimed_this_stop = False
            self.last_node_id = current_node_id
            self.visited_node_ids.add(current_node_id)

        # Update graph if edges are provided
        if inquire.edges:
            if self.start_msg:
                self.graph = MapGraph(self.start_msg.nodes, inquire.edges)
            else:
                self.graph = MapGraph(inquire.nodes, inquire.edges)

        # Update process_nodes from inquire.nodes[] (runtime state may override)
        for node in inquire.nodes:
            nid = node.get("nodeId", "")
            pt = node.get("processType")
            if nid and pt:
                self.process_nodes[nid] = {
                    "processType": pt,
                    "processRound": node.get("processRound", 0),
                }
            if nid and (nid in self.guard_blocked_targets or nid in self.avoid_route_nodes):
                guard = node.get("guard", {}) or {}
                owner_team = guard.get("ownerTeamId") or guard.get("teamId", "")
                owner_player = guard.get("playerId")
                is_enemy_active = (
                    guard.get("active", True) is not False
                    and guard.get("defense", 0) > 0
                    and (
                        (owner_team and owner_team != player.get("teamId", ""))
                        or (owner_player is not None and owner_player != self.player_id)
                    )
                )
                if not is_enemy_active:
                    self.guard_blocked_targets.discard(nid)
                    logger.info("Round %d: Guard at %s no longer blocks, clearing route block", inquire.round, nid)
                elif current_node_id and self.graph and nid in self.graph.get_neighbors(current_node_id):
                    self.guard_blocked_targets.add(nid)

        if current_node_id and self.graph:
            for node in inquire.nodes:
                nid = node.get("nodeId", "")
                if not nid or nid not in self.graph.get_neighbors(current_node_id):
                    continue
                guard = node.get("guard", {}) or {}
                owner_team = guard.get("ownerTeamId") or guard.get("teamId", "")
                owner_player = guard.get("playerId")
                is_enemy_active = (
                    guard.get("active", True) is not False
                    and guard.get("defense", 0) > 0
                    and (
                        (owner_team and owner_team != player.get("teamId", ""))
                        or (owner_player is not None and owner_player != self.player_id)
                    )
                )
                if is_enemy_active:
                    self.guard_blocked_targets.add(nid)

        # Check last action result
        last_failed = False
        last_error = ""
        for ar in inquire.action_results:
            if ar.get("playerId") == self.player_id:
                if ar.get("accepted") is False:
                    last_failed = True
                    last_error = ar.get("errorCode", "")
                    logger.info("Round %d: Last action rejected: %s", inquire.round, last_error)
                    # TARGET_NOT_REACHABLE 可能由障碍/设卡引起，不永久删边
                    if last_error == "INVALID_ACTION_TYPE" and ar.get("action") == "RUSH_SPEED":
                        self.rush_speed_failed = True
                        logger.info("Round %d: RUSH_SPEED rejected as INVALID_ACTION_TYPE, disabling", inquire.round)
                    # Track CLAIM_TASK business rejections that should not be retried.
                    # Note: actionResults doesn't include taskId, use last_claimed_task_id
                    if (
                        ar.get("action") == "CLAIM_TASK"
                        and last_error in {
                            "RESOURCE_NOT_ENOUGH",
                            "TASK_REQUIREMENT_NOT_MET",
                            "TASK_EXPIRED",
                            "WINDOW_DRAW_RETRY_LIMIT",
                        }
                    ):
                        failed_tid = self.last_claimed_task_id
                        if failed_tid:
                            self.failed_task_ids.add(failed_tid)
                            self.pending_task_hold_task_id = ""
                            self.pending_task_hold_node_id = ""
                            self.pending_task_hold_until_round = 0
                            logger.info("Round %d: Task %s rejected (%s), adding to failed list", inquire.round, failed_tid, last_error)
                    if ar.get("action") == "CLAIM_TASK" and last_error == "OBJECT_BUSY":
                        failed_tid = self.last_claimed_task_id
                        if failed_tid and failed_tid in enemy_busy_task_ids:
                            logger.info(
                                "Round %d: Task %s busy (enemy processing), skip hold",
                                inquire.round, failed_tid,
                            )
                        else:
                            self.pending_task_hold_task_id = self.last_claimed_task_id
                            self.pending_task_hold_node_id = self.last_claimed_task_node_id or current_node_id or ""
                            self.pending_task_hold_until_round = inquire.round + 6
                            logger.info(
                                "Round %d: Task %s busy at %s, holding until round %d",
                                inquire.round, self.last_claimed_task_id,
                                self.pending_task_hold_node_id, self.pending_task_hold_until_round,
                            )
                    if last_error == "PROCESS_REQUIRED" and current_node_id:
                        self.processed_node_ids.discard(current_node_id)
                        logger.info("Round %d: PROCESS_REQUIRED at %s, clearing processed flag", inquire.round, current_node_id)
                    if (
                        (ar.get("action") == "FORCED_PASS" or self.last_forced_pass_target)
                        and last_error in {
                        "TARGET_NOT_FOUND",
                        "TARGET_NOT_REACHABLE",
                        "ACTION_REJECTED",
                        "FORCED_PASS_REPEAT",
                        "OBJECT_BUSY",
                        }
                    ):
                        target = ar.get("targetNodeId", "") or self.last_forced_pass_target
                        if target:
                            self.forced_pass_failed_targets.add(target)
                            logger.info("Round %d: FORCED_PASS %s rejected (%s), will try normal move", inquire.round, target, last_error)
                    if last_error == "MOVE_BLOCKED_BY_GUARD":
                        target = ar.get("targetNodeId") or player.get("nextNodeId", "")
                        if target:
                            self.guard_blocked_targets.add(target)
                            logger.info("Round %d: Guard blocks %s, will reroute/break", inquire.round, target)
                    if last_error == "MOVING_ACTION_FORBIDDEN":
                        target = ar.get("targetNodeId") or player.get("nextNodeId", "")
                        if target:
                            self.guard_blocked_targets.add(target)
                            logger.info(
                                "Round %d: MOVING_ACTION_FORBIDDEN at %s, guard tax in progress",
                                inquire.round, target,
                            )

        # Sync squad clear pending with live obstacle state
        for node in inquire.nodes:
            nid = node.get("nodeId", "")
            if nid in self.squad_clear_pending and not node.get("hasObstacle", False):
                self.squad_clear_pending.discard(nid)

        # Also check events for rejections and cache contest info
        for ev in inquire.events:
            ev_type = ev.get("type", "")
            payload = ev.get("payload", {})
            if ev_type == "ACTION_REJECTED" and payload.get("playerId") == self.player_id:
                last_error = payload.get("errorCode", last_error)
                last_failed = True
                if last_error == "INVALID_ACTION_TYPE" and payload.get("action") == "RUSH_SPEED":
                    self.rush_speed_failed = True
                    logger.info("Round %d: RUSH_SPEED INVALID_ACTION_TYPE (from event), disabling", inquire.round)
                # Track CLAIM_TASK business rejections from events.
                if (
                    payload.get("action") == "CLAIM_TASK"
                    and last_error in {
                        "RESOURCE_NOT_ENOUGH",
                        "TASK_REQUIREMENT_NOT_MET",
                        "TASK_EXPIRED",
                        "WINDOW_DRAW_RETRY_LIMIT",
                    }
                ):
                    failed_tid = self.last_claimed_task_id
                    if failed_tid:
                        self.failed_task_ids.add(failed_tid)
                        self.pending_task_hold_task_id = ""
                        self.pending_task_hold_node_id = ""
                        self.pending_task_hold_until_round = 0
                        logger.info("Round %d: Task %s %s (from event), adding to failed list", inquire.round, failed_tid, last_error)
                if payload.get("action") == "CLAIM_TASK" and last_error == "OBJECT_BUSY":
                    failed_tid = self.last_claimed_task_id
                    if failed_tid and failed_tid in enemy_busy_task_ids:
                        logger.info(
                            "Round %d: Task %s busy (enemy processing, from event), skip hold",
                            inquire.round, failed_tid,
                        )
                    else:
                        self.pending_task_hold_task_id = self.last_claimed_task_id
                        self.pending_task_hold_node_id = self.last_claimed_task_node_id or current_node_id or ""
                        self.pending_task_hold_until_round = inquire.round + 6
                        logger.info(
                            "Round %d: Task %s busy at %s (from event), holding until round %d",
                            inquire.round, self.last_claimed_task_id,
                            self.pending_task_hold_node_id, self.pending_task_hold_until_round,
                        )
                if last_error == "PROCESS_REQUIRED" and current_node_id:
                    self.processed_node_ids.discard(current_node_id)
                if (
                    (payload.get("action") == "FORCED_PASS" or self.last_forced_pass_target)
                    and last_error in {
                    "TARGET_NOT_FOUND",
                    "TARGET_NOT_REACHABLE",
                    "ACTION_REJECTED",
                    "FORCED_PASS_REPEAT",
                    "OBJECT_BUSY",
                    }
                ):
                    target = payload.get("targetNodeId", "") or self.last_forced_pass_target
                    if target:
                        self.forced_pass_failed_targets.add(target)
                        logger.info("Round %d: FORCED_PASS %s rejected from event (%s), will try normal move", inquire.round, target, last_error)
                if last_error == "MOVE_BLOCKED_BY_GUARD":
                    target = payload.get("targetNodeId") or player.get("nextNodeId", "")
                    if target:
                        self.guard_blocked_targets.add(target)
                if last_error == "MOVING_ACTION_FORBIDDEN":
                    target = payload.get("targetNodeId") or player.get("nextNodeId", "")
                    if target:
                        self.guard_blocked_targets.add(target)
                        logger.info(
                            "Round %d: MOVING_ACTION_FORBIDDEN at %s (from event), guard tax in progress",
                            inquire.round, target,
                        )
            if ev_type == "GUARD_BREAK":
                node_id = payload.get("nodeId") or payload.get("targetNodeId", "")
                if node_id:
                    self.guard_blocked_targets.discard(node_id)
                    self.forced_pass_failed_targets.discard(node_id)
                    logger.info("Round %d: Guard broken at %s", inquire.round, node_id)
                    if payload.get("ownerTeamId") == player.get("teamId", "") or payload.get("playerId") == self.player_id:
                        self.own_guard_sites.discard(node_id)
            if ev_type == "OBSTACLE_CLEAR":
                node_id = payload.get("nodeId") or payload.get("targetNodeId", "")
                if node_id:
                    self.squad_clear_pending.discard(node_id)
            if ev_type == "GUARD_WEATHERING":
                node_id = payload.get("nodeId", "")
                if node_id and payload.get("defense", 1) <= 0:
                    self.guard_blocked_targets.discard(node_id)
                    self.forced_pass_failed_targets.discard(node_id)
                    self.own_guard_sites.discard(node_id)
            if ev_type in ("PROCESS_COMPLETE", "VERIFY_GATE_COMPLETE"):
                if payload.get("playerId") == self.player_id:
                    node_id = payload.get("nodeId") or payload.get("targetNodeId")
                    if node_id:
                        self.processed_node_ids.add(node_id)
                        logger.debug("Round %d: Process complete at %s", inquire.round, node_id)
            # Cache contest ID when window contest starts
            if ev_type == "WINDOW_CONTEST_START":
                cid = payload.get("contestId", "")
                if cid:
                    self.active_contest_id = cid
                    logger.info("Round %d: Window contest started: %s", inquire.round, cid)
            # Clear cached contest ID when contest ends
            if ev_type in ("WINDOW_CONTEST_END", "WINDOW_CONTEST_DRAW"):
                cid = payload.get("contestId", "")
                if cid and self.active_contest_id == cid:
                    self.active_contest_id = ""
                    logger.info("Round %d: Window contest ended: %s", inquire.round, cid)

        # Track how long we are stuck on a guarded hop (next hop or blocked neighbor)
        player_state = player.get("state", "")
        next_nid = player.get("nextNodeId", "")
        stuck_block = ""
        if next_nid and next_nid in self.guard_blocked_targets:
            stuck_block = next_nid
        elif current_node_id and self.graph:
            for tgt in self.guard_blocked_targets:
                if tgt in self.avoid_route_nodes:
                    continue
                if tgt in self.graph.get_neighbors(current_node_id):
                    stuck_block = tgt
                    break
        if (
            player_state in ("WAITING", "MOVING", "IDLE")
            and stuck_block
        ):
            if stuck_block == self.guard_stuck_target and current_node_id == self.last_node_id:
                self.guard_stuck_rounds += 1
            else:
                self.guard_stuck_target = stuck_block
                self.guard_stuck_rounds = 1
            if self.guard_stuck_rounds >= GUARD_STUCK_AVOID_ROUNDS:
                if stuck_block not in self.avoid_route_nodes:
                    logger.info(
                        "Round %d: Permanently avoiding %s after %d stuck rounds",
                        inquire.round, stuck_block, self.guard_stuck_rounds,
                    )
                self.avoid_route_nodes.add(stuck_block)
                self.guard_blocked_targets.discard(stuck_block)
        elif not stuck_block:
            self.guard_stuck_target = ""
            self.guard_stuck_rounds = 0

        # Determine gate and terminal IDs (from start message or inquire nodes)
        gate_node_id = ""
        terminal_node_ids: list[str] = []
        if self.start_msg:
            gate_node_id = self.start_msg.gate_node_id
            terminal_node_ids = self.start_msg.terminal_node_ids
        # Fallback: scan inquire nodes for gate/terminal markers
        if not gate_node_id:
            for node in inquire.nodes:
                if node.get("gateNodeId") or node.get("nodeType") == "GATE":
                    gate_node_id = node.get("nodeId", "")
                    break
        if not terminal_node_ids:
            for node in inquire.nodes:
                if node.get("terminalNodeId") or node.get("nodeType") in ("TERMINAL", "FINISH") or node.get("terminal"):
                    terminal_node_ids.append(node.get("nodeId", ""))

        # Decide action
        action_msg = decide_action(
            self.match_id,
            inquire.round,
            self.player_id,
            player,
            self.graph,
            current_node=current_node,
            process_nodes=self.process_nodes,
            contests=inquire.contests,
            events=inquire.events,
            active_contest_id=self.active_contest_id,
            last_move_failed=last_failed,
            last_move_error=last_error,
            gate_node_id=gate_node_id,
            terminal_node_ids=terminal_node_ids,
            tasks=inquire.tasks,
            phase=inquire.phase,
            processed_node_ids=self.processed_node_ids,
            visited_node_ids=self.visited_node_ids,
            weather=inquire.weather,
            all_players=inquire.players,
            inquire_nodes=inquire.nodes,
            failed_task_ids=self.failed_task_ids,
            rush_speed_failed=self.rush_speed_failed,
            guard_blocked_targets=self.guard_blocked_targets,
            avoid_route_nodes=self.avoid_route_nodes,
            pending_task_hold_task_id=self.pending_task_hold_task_id,
            pending_task_hold_node_id=self.pending_task_hold_node_id,
            pending_task_hold_until_round=self.pending_task_hold_until_round,
            forced_pass_failed_targets=self.forced_pass_failed_targets,
            squad_clear_pending=self.squad_clear_pending,
            guard_stuck_rounds=self.guard_stuck_rounds,
            guard_stuck_target=self.guard_stuck_target,
            own_guard_sites=self.own_guard_sites,
            map_gameplay=self.map_gameplay,
            task_claimed_this_stop=self.task_claimed_this_stop,
        )

        self.send_message(action_msg)

        # Track action counts
        actions = action_msg.get("msg_data", {}).get("actions", [])
        action_type = actions[0].get("action", "") if actions else "EMPTY"
        action_detail = ""
        if action_type == "MOVE":
            action_detail = f"->{actions[0].get('targetNodeId', '?')}"
        elif action_type == "FORCED_PASS":
            action_detail = f"->{actions[0].get('targetNodeId', '?')}"
        elif action_type == "CLAIM_RESOURCE":
            action_detail = f"({actions[0].get('resourceType', '?')})"
        elif action_type == "CLAIM_TASK":
            action_detail = f"({actions[0].get('taskId', '?')})"
            self.last_claimed_task_id = actions[0].get("taskId", "")
            self.last_claimed_task_node_id = current_node_id or ""
            self.task_claimed_this_stop = True
        self.last_forced_pass_target = actions[0].get("targetNodeId", "") if action_type == "FORCED_PASS" else ""
        for action_item in actions:
            if action_item.get("action") == "SQUAD_CLEAR":
                target = action_item.get("targetNodeId", "")
                if target:
                    self.squad_clear_pending.add(target)
            if action_item.get("action") == "SET_GUARD":
                target = action_item.get("targetNodeId", "")
                if target:
                    self.avoid_route_nodes.add(target)
                    self.own_guard_sites.add(target)
                    logger.info(
                        "Round %d: Avoid own guard node %s after SET_GUARD (advance separately)",
                        inquire.round, target,
                    )
        if action_type == "MOVE":
            self.move_count += 1
        elif action_type in ("PROCESS", "DOCK", "VERIFY_GATE"):
            self.process_count += 1

        # Log state periodically (every 10 rounds or on important events)
        if inquire.round % 10 == 0 or last_failed or action_type in ("MOVE", "PROCESS", "DOCK", "VERIFY_GATE", "CLAIM_RESOURCE", "CLAIM_TASK"):
            logger.info(
                "Round %d: state=%s node=%s action=%s%s (moves:%d process:%d)%s",
                inquire.round,
                player.get("state", "?"),
                current_node_id,
                action_type,
                action_detail,
                self.move_count,
                self.process_count,
                f" rejected={last_error}" if last_failed else "",
            )

        return action_msg

    def handle_over(self, over: OverMessage) -> None:
        """Handle over message."""
        logger.info("Game over: %s, winner=%s, rounds=%d",
                     over.result_type, over.winner_player_id, over.over_round)
        self.running = False

    def run(self) -> None:
        """Main game loop: connect, register, and process messages."""
        self.connect()
        self.send_registration()
        self.running = True

        while self.running:
            messages = self.receive_messages()
            for raw_msg in messages:
                msg = parse_message(raw_msg)

                if isinstance(msg, StartMessage):
                    self.handle_start(msg)
                    self.send_ready()

                elif isinstance(msg, InquireMessage):
                    self.handle_inquire(msg)

                elif isinstance(msg, OverMessage):
                    self.handle_over(msg)

                else:
                    # error or unknown message
                    msg_name = raw_msg.get("msg_name", "?")
                    if msg_name == "error":
                        error_data = raw_msg.get("msg_data", {})
                        logger.warning("Server error: %s - %s (raw: %s)",
                                       error_data.get("errorCode", ""),
                                       error_data.get("message", ""),
                                       str(raw_msg)[:200])

        if self.sock:
            self.sock.close()
            logger.info("Connection closed. Total rounds: %d, moves: %d, process: %d",
                        self.round_count, self.move_count, self.process_count)