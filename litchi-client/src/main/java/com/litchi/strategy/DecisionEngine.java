package com.litchi.strategy;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.litchi.action.ActionBuilder;
import com.litchi.game.GameContext;
import com.litchi.game.GameMap;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class DecisionEngine {

    private static final String[] RESOURCE_PRIORITY = {
            "FAST_HORSE", "SHORT_HORSE", "ICE_BOX", "OFFICIAL_PERMIT", "PASS_TOKEN", "INTEL", "BOAT_RIGHT"
    };

    public JsonArray decide(GameContext context) {
        if (context.isDelivered()) {
            return ActionBuilder.single(ActionBuilder.of("WAIT"));
        }

        JsonObject windowAction = decideWindow(context);
        if (windowAction != null) {
            return ActionBuilder.single(windowAction);
        }

        String state = context.getState();

        if ("RESTING".equals(state) || "FORCED_PASSING".equals(state) || "VERIFYING".equals(state)
                || "PROCESSING".equals(state) || "CONTESTING".equals(state)) {
            return heartbeat();
        }
        if ("MOVING".equals(state) || "WAITING".equals(state)) {
            return decideWhileMoving(context);
        }
        return decideWhileIdle(context);
    }

    private JsonArray heartbeat() {
        return new JsonArray();
    }

    private JsonObject decideWindow(GameContext context) {
        JsonArray contests = context.getContests();
        for (JsonElement element : contests) {
            JsonObject contest = element.getAsJsonObject();
            if (contest.has("status") && "SUPPRESSED".equals(contest.get("status").getAsString())) {
                continue;
            }
            if (contest.has("resolved") && contest.get("resolved").getAsBoolean()) {
                continue;
            }
            if (!contest.has("contestId")) {
                continue;
            }
            String contestId = contest.get("contestId").getAsString();
            String card = chooseWindowCard(context, contest);
            return ActionBuilder.windowCard(contestId, card);
        }
        return null;
    }

    private String chooseWindowCard(GameContext context, JsonObject contest) {
        String type = contest.has("contestType") ? contest.get("contestType").getAsString() : "";
        if ("GATE".equals(type) || "RESOURCE".equals(type) || "TASK".equals(type) || "DOCK".equals(type)) {
            if (context.getGuardActionPoint() > 0) {
                return "BING_ZHENG";
            }
            if (context.getResourceCount("PASS_TOKEN") > 0 || context.getResourceCount("OFFICIAL_PERMIT") > 0) {
                return "YAN_DIE";
            }
            if (context.getFreshness() >= 80 && context.getGoodFruit() > 0) {
                return "XIAN_GONG";
            }
        }
        if ("PASS".equals(type) && context.getResourceCount("FAST_HORSE") + context.getResourceCount("SHORT_HORSE") > 0) {
            return "QIANG_XING";
        }
        return "ABSTAIN";
    }

    private JsonArray decideWhileMoving(GameContext context) {
        JsonObject horse = chooseHorseUse(context);
        if (horse != null) {
            return ActionBuilder.single(horse);
        }
        return heartbeat();
    }

    private JsonArray decideWhileIdle(GameContext context) {
        String currentNode = context.getCurrentNodeId();
        GameMap map = context.getMap();

        JsonObject deliver = tryDeliver(context);
        if (deliver != null) {
            return withSquad(context, deliver, chooseSquadAction(context, currentNode));
        }

        JsonObject verify = tryVerifyGate(context, currentNode);
        if (verify != null) {
            return withSquad(context, verify, chooseSquadAction(context, currentNode));
        }

        JsonObject rush = tryRushTactic(context, currentNode);
        if (rush != null) {
            return withSquad(context, rush, chooseSquadAction(context, currentNode));
        }

        JsonObject resourceUse = chooseResourceUse(context, currentNode);
        if (resourceUse != null) {
            return withSquad(context, resourceUse, chooseSquadAction(context, currentNode));
        }

        JsonObject process = tryProcess(context, currentNode, map);
        if (process != null) {
            return withSquad(context, process, chooseSquadAction(context, currentNode));
        }

        JsonObject task = tryClaimTask(context, currentNode);
        if (task != null) {
            return withSquad(context, task, chooseSquadAction(context, currentNode));
        }

        JsonObject resource = tryClaimResource(context, currentNode);
        if (resource != null) {
            return withSquad(context, resource, chooseSquadAction(context, currentNode));
        }

        JsonObject move = tryMoveTowardsGoal(context, currentNode);
        if (move != null) {
            return withSquad(context, move, chooseSquadAction(context, currentNode));
        }

        return withSquad(context, ActionBuilder.of("WAIT"), chooseSquadAction(context, currentNode));
    }

    private JsonArray withSquad(GameContext context, JsonObject mainAction, JsonObject squadAction) {
        if (squadAction == null) {
            return ActionBuilder.single(mainAction);
        }
        return ActionBuilder.pair(mainAction, squadAction);
    }

    private JsonObject tryDeliver(GameContext context) {
        if (!mapEquals(context.getCurrentNodeId(), context.getMap().getTerminalNodeId())) {
            return null;
        }
        if (!context.isVerified() || context.getGoodFruit() <= 0 || context.getFreshness() <= 0) {
            return null;
        }
        return ActionBuilder.deliver();
    }

    private JsonObject tryVerifyGate(GameContext context, String currentNode) {
        if (!context.isRushPhase() || context.isVerified()) {
            return null;
        }
        if (!mapEquals(currentNode, context.getMap().getGateNodeId())) {
            return null;
        }
        String rushTactic = canUseBreakOrder(context) ? "BREAK_ORDER" : null;
        return ActionBuilder.verifyGate(context.getMap().getGateNodeId(), rushTactic);
    }

    private JsonObject tryRushTactic(GameContext context, String currentNode) {
        if (!context.isRushPhase()) {
            return null;
        }
        JsonObject self = context.getSelfPlayer();
        if (self != null && self.has("rushTacticUsedCount") && self.get("rushTacticUsedCount").getAsInt() > 0) {
            return null;
        }
        if (context.getFreshness() < 45) {
            return ActionBuilder.rushProtect();
        }
        if ("MOVING".equals(context.getState()) || context.getResourceCount("FAST_HORSE") > 0 || context.getResourceCount("SHORT_HORSE") > 0) {
            return null;
        }
        if (context.getGoodFruit() >= 2) {
            return ActionBuilder.rushSpeed();
        }
        return null;
    }

    private boolean canUseBreakOrder(GameContext context) {
        JsonObject self = context.getSelfPlayer();
        if (self == null) {
            return false;
        }
        int used = self.has("rushTacticUsedCount") ? self.get("rushTacticUsedCount").getAsInt() : 0;
        if (used > 0) {
            return false;
        }
        int badFruit = self.has("badFruit") ? self.get("badFruit").getAsInt() : 0;
        return badFruit >= 2 || context.getGoodFruit() > 0;
    }

    private JsonObject chooseResourceUse(GameContext context, String currentNode) {
        if (context.getFreshness() < 72 && context.getResourceCount("ICE_BOX") > 0) {
            return ActionBuilder.useResource("ICE_BOX", null);
        }
        JsonObject horse = chooseHorseUse(context);
        if (horse != null) {
            return horse;
        }
        String scoutTarget = nextStrategicNode(context, currentNode);
        if (scoutTarget != null && context.getResourceCount("INTEL") > 0 && pathDistance(context, currentNode, scoutTarget) <= 15) {
            return ActionBuilder.useResource("INTEL", scoutTarget);
        }
        return null;
    }

    private JsonObject chooseHorseUse(GameContext context) {
        if (context.getResourceCount("FAST_HORSE") > 0) {
            return ActionBuilder.useResource("FAST_HORSE", null);
        }
        if (context.getResourceCount("SHORT_HORSE") > 0) {
            return ActionBuilder.useResource("SHORT_HORSE", null);
        }
        return null;
    }

    private JsonObject tryProcess(GameContext context, String currentNode, GameMap map) {
        if (context.hasProcessDone(currentNode)) {
            return null;
        }
        JsonObject nodeState = context.findNodeState(currentNode);
        GameMap.NodeInfo nodeInfo = map.getNode(currentNode);
        String processType = null;
        if (nodeState != null && nodeState.has("processType") && !nodeState.get("processType").isJsonNull()) {
            processType = nodeState.get("processType").getAsString();
        } else if (nodeInfo != null) {
            processType = nodeInfo.getProcessType();
        }
        if (processType == null || processType.trim().isEmpty()) {
            return null;
        }
        if ("VERIFY".equals(processType)) {
            return null;
        }
        if ("BOARD".equals(processType)) {
            JsonObject dock = ActionBuilder.of("DOCK");
            dock.addProperty("targetNodeId", currentNode);
            return dock;
        }
        return ActionBuilder.process(currentNode);
    }

    private JsonObject tryClaimTask(GameContext context, String currentNode) {
        List<JsonObject> candidates = new ArrayList<>();
        for (JsonElement element : context.getTasks()) {
            JsonObject task = element.getAsJsonObject();
            if (!task.has("active") || !task.get("active").getAsBoolean()) {
                continue;
            }
            if (task.has("completed") && task.get("completed").getAsBoolean()) {
                continue;
            }
            if (task.has("failed") && task.get("failed").getAsBoolean()) {
                continue;
            }
            if (task.has("protectionPlayerId") && task.get("protectionPlayerId").getAsInt() != 0
                    && task.get("protectionPlayerId").getAsInt() != context.getPlayerId()) {
                continue;
            }
            String nodeId = task.get("nodeId").getAsString();
            String templateId = task.get("taskTemplateId").getAsString();
            if ("T04".equals(templateId)) {
                java.util.Set<String> emptyBlocked = new java.util.HashSet<String>();
                if (nodeId.equals(currentNode) || context.getMap().neighbors(currentNode, emptyBlocked).contains(nodeId)) {
                    candidates.add(task);
                }
                continue;
            }
            if (!nodeId.equals(currentNode)) {
                continue;
            }
            if ("T06".equals(templateId)
                    && context.getResourceCount("FAST_HORSE") == 0
                    && context.getResourceCount("SHORT_HORSE") == 0) {
                continue;
            }
            candidates.add(task);
        }
        candidates.sort(Comparator.comparingInt((JsonObject task) -> -task.get("score").getAsInt())
                .thenComparingInt(task -> task.get("processRound").getAsInt()));
        if (candidates.isEmpty()) {
            return null;
        }
        return ActionBuilder.claimTask(candidates.get(0).get("taskId").getAsString());
    }

    private JsonObject tryClaimResource(GameContext context, String currentNode) {
        Map<String, Integer> stock = context.nodeResourceStock(currentNode);
        for (String resourceType : RESOURCE_PRIORITY) {
            if (stock.getOrDefault(resourceType, 0) > 0) {
                return ActionBuilder.claimResource(currentNode, resourceType);
            }
        }
        return null;
    }

    private JsonObject tryMoveTowardsGoal(GameContext context, String currentNode) {
        String goal = chooseGoalNode(context);
        Set<String> blocked = context.blockedNodes();
        blocked.remove(currentNode);
        List<String> path = context.getMap().shortestPath(currentNode, goal, blocked);
        if (path.size() < 2) {
            java.util.Set<String> emptyBlocked = new java.util.HashSet<String>();
            path = context.getMap().shortestPath(currentNode, goal, emptyBlocked);
        }
        if (path.size() < 2) {
            return null;
        }
        String nextNode = path.get(1);
        java.util.Set<String> emptyBlocked = new java.util.HashSet<String>();
        if (!context.getMap().neighbors(currentNode, emptyBlocked).contains(nextNode)) {
            return null;
        }
        return ActionBuilder.move(nextNode);
    }

    private String chooseGoalNode(GameContext context) {
        if (context.isRushPhase()) {
            if (!context.isVerified()) {
                return context.getMap().getGateNodeId();
            }
            return context.getMap().getTerminalNodeId();
        }
        return context.getMap().getGateNodeId();
    }

    private JsonObject chooseSquadAction(GameContext context, String currentNode) {
        if (context.isRushPhase() || context.getSquadAvailable() < 1) {
            return null;
        }
        String scoutTarget = nextStrategicNode(context, currentNode);
        if (scoutTarget != null) {
            return ActionBuilder.squadScout(scoutTarget);
        }
        return null;
    }

    private String nextStrategicNode(GameContext context, String currentNode) {
        String goal = chooseGoalNode(context);
        List<String> path = context.getMap().shortestPath(currentNode, goal, context.blockedNodes());
        if (path.size() < 2) {
            java.util.Set<String> emptyBlocked = new java.util.HashSet<String>();
            path = context.getMap().shortestPath(currentNode, goal, emptyBlocked);
        }
        if (path.size() < 2) {
            return null;
        }
        for (int i = 1; i < path.size(); i++) {
            String nodeId = path.get(i);
            GameMap.NodeInfo info = context.getMap().getNode(nodeId);
            if (info != null && info.getProcessType() != null && !info.getProcessType().trim().isEmpty()) {
                return nodeId;
            }
        }
        return path.get(Math.min(2, path.size() - 1));
    }

    private int pathDistance(GameContext context, String from, String to) {
        List<String> path = context.getMap().shortestPath(from, to, new java.util.HashSet<String>());
        if (path.size() < 2) {
            return Integer.MAX_VALUE;
        }
        int total = 0;
        for (int i = 0; i < path.size() - 1; i++) {
            GameMap.Edge edge = context.getMap().findEdge(path.get(i), path.get(i + 1));
            if (edge != null) {
                total += edge.getDistance();
            }
        }
        return total;
    }

    private boolean mapEquals(String left, String right) {
        return left != null && right != null && left.equalsIgnoreCase(right);
    }
}
