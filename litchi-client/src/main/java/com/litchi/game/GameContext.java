package com.litchi.game;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class GameContext {

    private final int playerId;
    private final GameMap map = new GameMap();

    private String matchId;
    private String teamId;
    private int durationRound = 600;
    private int currentRound;
    private String phase = "NORMAL";
    private JsonObject lastInquireData;
    private JsonObject selfPlayer;
    private JsonArray tasks = new JsonArray();
    private JsonArray nodes = new JsonArray();
    private JsonArray contests = new JsonArray();
    private final Set<String> completedProcessNodes = new HashSet<>();

    public GameContext(int playerId) {
        this.playerId = playerId;
    }

    public void onStart(JsonObject startData) {
        matchId = startData.get("matchId").getAsString();
        if (startData.has("durationRound")) {
            durationRound = startData.get("durationRound").getAsInt();
        }
        map.loadFromStart(startData);
        if (startData.has("players") && startData.get("players").isJsonArray()) {
            for (JsonElement element : startData.getAsJsonArray("players")) {
                JsonObject player = element.getAsJsonObject();
                if (player.get("playerId").getAsInt() == playerId) {
                    teamId = player.get("teamId").getAsString();
                    break;
                }
            }
        }
    }

    public void onInquire(JsonObject inquireData) {
        lastInquireData = inquireData;
        currentRound = inquireData.get("round").getAsInt();
        if (inquireData.has("phase")) {
            phase = inquireData.get("phase").getAsString();
        }
        if (inquireData.has("edges") && inquireData.get("edges").isJsonArray()) {
            map.updateEdges(inquireData.getAsJsonArray("edges"));
        }
        if (inquireData.has("players") && inquireData.get("players").isJsonArray()) {
            for (JsonElement element : inquireData.getAsJsonArray("players")) {
                JsonObject player = element.getAsJsonObject();
                if (player.get("playerId").getAsInt() == playerId) {
                    selfPlayer = player;
                    break;
                }
            }
        }
        tasks = inquireData.has("tasks") && inquireData.get("tasks").isJsonArray()
                ? inquireData.getAsJsonArray("tasks") : new JsonArray();
        nodes = inquireData.has("nodes") && inquireData.get("nodes").isJsonArray()
                ? inquireData.getAsJsonArray("nodes") : new JsonArray();
        contests = inquireData.has("contests") && inquireData.get("contests").isJsonArray()
                ? inquireData.getAsJsonArray("contests") : new JsonArray();
        trackProcessCompletion(inquireData);
    }

    private void trackProcessCompletion(JsonObject inquireData) {
        if (!inquireData.has("events") || !inquireData.get("events").isJsonArray()) {
            return;
        }
        for (JsonElement element : inquireData.getAsJsonArray("events")) {
            JsonObject event = element.getAsJsonObject();
            if (!"PROCESS_COMPLETE".equals(event.get("type").getAsString())) {
                continue;
            }
            JsonObject payload = event.getAsJsonObject("payload");
            if (payload.has("playerId") && payload.get("playerId").getAsInt() == playerId && payload.has("nodeId")) {
                completedProcessNodes.add(payload.get("nodeId").getAsString());
            }
        }
    }

    public int getPlayerId() {
        return playerId;
    }

    public String getMatchId() {
        return matchId;
    }

    public String getTeamId() {
        return teamId;
    }

    public int getDurationRound() {
        return durationRound;
    }

    public int getCurrentRound() {
        return currentRound;
    }

    public String getPhase() {
        return phase;
    }

    public GameMap getMap() {
        return map;
    }

    public JsonObject getSelfPlayer() {
        return selfPlayer;
    }

    public JsonArray getTasks() {
        return tasks;
    }

    public JsonArray getNodes() {
        return nodes;
    }

    public JsonArray getContests() {
        return contests;
    }

    public JsonObject getLastInquireData() {
        return lastInquireData;
    }

    public boolean isRushPhase() {
        return "RUSH".equals(phase);
    }

    public String getState() {
        return selfPlayer != null && selfPlayer.has("state") ? selfPlayer.get("state").getAsString() : "IDLE";
    }

    public String getCurrentNodeId() {
        return selfPlayer != null && selfPlayer.has("currentNodeId") && !selfPlayer.get("currentNodeId").isJsonNull()
                ? selfPlayer.get("currentNodeId").getAsString() : map.getStartNodeId();
    }

    public String getNextNodeId() {
        return selfPlayer != null && selfPlayer.has("nextNodeId") && !selfPlayer.get("nextNodeId").isJsonNull()
                ? selfPlayer.get("nextNodeId").getAsString() : null;
    }

    public boolean isVerified() {
        return selfPlayer != null && selfPlayer.has("verified") && selfPlayer.get("verified").getAsBoolean();
    }

    public boolean isDelivered() {
        return selfPlayer != null && selfPlayer.has("delivered") && selfPlayer.get("delivered").getAsBoolean();
    }

    public int getGoodFruit() {
        return selfPlayer != null && selfPlayer.has("goodFruit") ? selfPlayer.get("goodFruit").getAsInt() : 0;
    }

    public double getFreshness() {
        return selfPlayer != null && selfPlayer.has("freshness") ? selfPlayer.get("freshness").getAsDouble() : 100.0;
    }

    public int getResourceCount(String resourceType) {
        if (selfPlayer == null || !selfPlayer.has("resources") || !selfPlayer.get("resources").isJsonObject()) {
            return 0;
        }
        JsonObject resources = selfPlayer.getAsJsonObject("resources");
        return resources.has(resourceType) ? resources.get(resourceType).getAsInt() : 0;
    }

    public int getGuardActionPoint() {
        return selfPlayer != null && selfPlayer.has("guardActionPoint")
                ? selfPlayer.get("guardActionPoint").getAsInt() : 0;
    }

    public int getSquadAvailable() {
        return selfPlayer != null && selfPlayer.has("squadAvailable")
                ? selfPlayer.get("squadAvailable").getAsInt() : 0;
    }

    public boolean hasProcessDone(String nodeId) {
        return completedProcessNodes.contains(nodeId);
    }

    public JsonObject findNodeState(String nodeId) {
        for (JsonElement element : nodes) {
            JsonObject node = element.getAsJsonObject();
            if (nodeId.equals(node.get("nodeId").getAsString())) {
                return node;
            }
        }
        return null;
    }

    public Set<String> blockedNodes() {
        Set<String> blocked = new HashSet<>();
        String opponentTeam = "RED".equals(teamId) ? "BLUE" : "RED";
        for (JsonElement element : nodes) {
            JsonObject node = element.getAsJsonObject();
            String nodeId = node.get("nodeId").getAsString();
            if (node.has("hasObstacle") && node.get("hasObstacle").getAsBoolean()) {
                blocked.add(nodeId);
            }
            if (node.has("guard") && node.get("guard").isJsonObject()) {
                JsonObject guard = node.getAsJsonObject("guard");
                if (guard.has("active") && guard.get("active").getAsBoolean()
                        && guard.has("ownerTeamId") && opponentTeam.equals(guard.get("ownerTeamId").getAsString())
                        && guard.has("defense") && guard.get("defense").getAsInt() > 0) {
                    blocked.add(nodeId);
                }
            }
        }
        return blocked;
    }

    public Map<String, Integer> nodeResourceStock(String nodeId) {
        Map<String, Integer> stock = new HashMap<>();
        JsonObject node = findNodeState(nodeId);
        if (node == null || !node.has("resourceStock") || !node.get("resourceStock").isJsonObject()) {
            return stock;
        }
        JsonObject resourceStock = node.getAsJsonObject("resourceStock");
        for (Map.Entry<String, JsonElement> entry : resourceStock.entrySet()) {
            stock.put(entry.getKey(), entry.getValue().getAsInt());
        }
        return stock;
    }
}
