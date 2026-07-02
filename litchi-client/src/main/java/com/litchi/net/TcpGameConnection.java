package com.litchi.net;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public final class TcpGameConnection implements AutoCloseable {

    private final Socket socket;
    private final InputStream input;
    private final OutputStream output;
    private final byte[] lengthBuffer = new byte[5];
    private final ByteArrayOutputStream pending = new ByteArrayOutputStream();

    public TcpGameConnection(String host, int port, int connectTimeoutMs) throws IOException {
        socket = new Socket();
        socket.connect(new InetSocketAddress(host, port), connectTimeoutMs);
        socket.setTcpNoDelay(true);
        input = socket.getInputStream();
        output = socket.getOutputStream();
    }

    public void send(JsonObject body) throws IOException {
        byte[] payload = body.toString().getBytes(StandardCharsets.UTF_8);
        if (payload.length > 99999) {
            throw new IOException("消息体超过 99999 字节");
        }
        String prefix = String.format("%05d", payload.length);
        output.write(prefix.getBytes(StandardCharsets.US_ASCII));
        output.write(payload);
        output.flush();
    }

    public JsonObject receive() throws IOException {
        while (true) {
            List<JsonObject> messages = drainBuffer();
            if (!messages.isEmpty()) {
                return messages.get(0);
            }
            int read = input.read(lengthBuffer);
            if (read < 0) {
                throw new IOException("连接已关闭");
            }
            if (read < 5) {
                throw new IOException("长度前缀不完整");
            }
            int bodyLength = Integer.parseInt(new String(lengthBuffer, StandardCharsets.US_ASCII));
            byte[] body = readFully(bodyLength);
            pending.write(body);
        }
    }

    private List<JsonObject> drainBuffer() {
        List<JsonObject> messages = new ArrayList<>();
        byte[] bytes = pending.toByteArray();
        int offset = 0;
        while (offset + 5 <= bytes.length) {
            int bodyLength;
            try {
                bodyLength = Integer.parseInt(new String(bytes, offset, 5, StandardCharsets.US_ASCII));
            } catch (NumberFormatException e) {
                break;
            }
            if (offset + 5 + bodyLength > bytes.length) {
                break;
            }
            String json = new String(bytes, offset + 5, bodyLength, StandardCharsets.UTF_8);
            messages.add(JsonParser.parseString(json).getAsJsonObject());
            offset += 5 + bodyLength;
        }
        pending.reset();
        if (offset < bytes.length) {
            pending.write(bytes, offset, bytes.length - offset);
        }
        return messages;
    }

    private byte[] readFully(int length) throws IOException {
        byte[] buffer = new byte[length];
        int offset = 0;
        while (offset < length) {
            int read = input.read(buffer, offset, length - offset);
            if (read < 0) {
                throw new IOException("消息体未读全");
            }
            offset += read;
        }
        return buffer;
    }

    @Override
    public void close() throws IOException {
        socket.close();
    }
}
