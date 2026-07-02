package com.litchi.config;

public final class ClientConfig {

    private final int playerId;
    private final String host;
    private final int port;
    private final String playerName;
    private final String version;

    public ClientConfig(int playerId, String host, int port, String playerName, String version) {
        this.playerId = playerId;
        this.host = host;
        this.port = port;
        this.playerName = playerName;
        this.version = version;
    }

    public static ClientConfig fromArgs(String[] args) {
        if (args.length < 3) {
            throw new IllegalArgumentException("用法: java -jar litchi-client.jar <playerId> <host> <port> [playerName] [version]");
        }
        int playerId = Integer.parseInt(args[0]);
        String host = args[1];
        int port = Integer.parseInt(args[2]);
        String playerName = args.length > 3 ? args[3] : "litchi-team-" + playerId;
        String version = args.length > 4 ? args[4] : "1.0";
        return new ClientConfig(playerId, host, port, playerName, version);
    }

    public int getPlayerId() {
        return playerId;
    }

    public String getHost() {
        return host;
    }

    public int getPort() {
        return port;
    }

    public String getPlayerName() {
        return playerName;
    }

    public String getVersion() {
        return version;
    }
}
