package com.litchi.game;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.litchi.config.ClientConfig;
import com.litchi.net.ProtocolMessages;
import com.litchi.net.TcpGameConnection;
import com.litchi.strategy.DecisionEngine;

public final class GameSession {

    private final ClientConfig config;
    private final GameContext context;
    private final DecisionEngine decisionEngine = new DecisionEngine();

    public GameSession(ClientConfig config) {
        this.config = config;
        this.context = new GameContext(config.getPlayerId());
    }

    public void run() throws Exception {
        System.out.printf("连接服务器 %s:%d，玩家 %d%n", config.getHost(), config.getPort(), config.getPlayerId());
        try (TcpGameConnection connection = new TcpGameConnection(config.getHost(), config.getPort(), 10000)) {
            connection.send(ProtocolMessages.registration(
                    config.getPlayerId(), config.getPlayerName(), config.getVersion()));
            System.out.println("已发送 registration");

            boolean readySent = false;
            while (true) {
                JsonObject message = connection.receive();
                String msgName = message.get("msg_name").getAsString();
                JsonObject msgData = message.getAsJsonObject("msg_data");

                switch (msgName) {
                    case "start":
                        handleStart(msgData);
                        break;
                    case "inquire":
                        if (!readySent) {
                            throw new IllegalStateException("未发送 ready 就收到 inquire");
                        }
                        handleInquire(connection, msgData);
                        break;
                    case "over":
                        handleOver(msgData);
                        return;
                    case "error":
                        handleError(msgData);
                        break;
                    default:
                        System.out.println("收到未处理消息: " + msgName);
                        break;
                }

                if ("start".equals(msgName) && !readySent) {
                    int round = msgData.has("round") ? msgData.get("round").getAsInt() : 1;
                    connection.send(ProtocolMessages.ready(context.getMatchId(), round, config.getPlayerId()));
                    readySent = true;
                    System.out.println("已发送 ready");
                }
            }
        }
    }

    private void handleStart(JsonObject startData) {
        context.onStart(startData);
        System.out.printf("开局 matchId=%s, team=%s, gate=%s, terminal=%s%n",
                context.getMatchId(),
                context.getTeamId(),
                context.getMap().getGateNodeId(),
                context.getMap().getTerminalNodeId());
    }

    private void handleInquire(TcpGameConnection connection, JsonObject inquireData) throws Exception {
        context.onInquire(inquireData);
        JsonArray actions = decisionEngine.decide(context);
        int round = inquireData.get("round").getAsInt();
        connection.send(ProtocolMessages.action(context.getMatchId(), round, config.getPlayerId(), actions));
        logDecision(round, actions);
    }

    private void handleOver(JsonObject overData) {
        System.out.println("比赛结束");
        if (overData.has("winnerPlayerId") && !overData.get("winnerPlayerId").isJsonNull()) {
            System.out.println("胜方 playerId=" + overData.get("winnerPlayerId").getAsInt());
        }
        if (overData.has("players") && overData.get("players").isJsonArray()) {
            overData.getAsJsonArray("players").forEach(element -> {
                JsonObject player = element.getAsJsonObject();
                System.out.printf("玩家 %d 总分=%d 已交付=%s%n",
                        player.get("playerId").getAsInt(),
                        player.has("totalScore") ? player.get("totalScore").getAsInt() : 0,
                        player.has("delivered") && player.get("delivered").getAsBoolean());
            });
        }
    }

    private void handleError(JsonObject errorData) {
        String code = errorData.has("errorCode") ? errorData.get("errorCode").getAsString() : "UNKNOWN";
        String message = errorData.has("message") ? errorData.get("message").getAsString() : "";
        System.out.printf("服务端 error: %s %s%n", code, message);
    }

    private void logDecision(int round, JsonArray actions) {
        if (actions.isEmpty()) {
            System.out.printf("[round %d] 心跳 actions=[]%n", round);
            return;
        }
        System.out.printf("[round %d] actions=%s%n", round, actions);
    }
}
