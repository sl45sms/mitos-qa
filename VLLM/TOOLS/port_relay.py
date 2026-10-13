#!/usr/bin/env python3
"""Tiny TCP relay: forward a compute-node port to the login node.

Why this exists: compute nodes (m01..m48) are reachable from the login node,
but NOT from your PC, and VS Code's Ports panel can only forward ports that
listen on the host it is connected to (login05). This relay bridges the two:
it listens on login05:127.0.0.1:<port> and forwards every incoming connection
to <node>:<port> on the cluster network.

Usage:
    python3 port_relay.py <target_host> [port] [listen_port]

Example - forward m42:8000 so it appears on the login node as localhost:8000:
    python3 port_relay.py m42 8000

Then in VS Code: Command Palette -> "Ports: Focus on Ports View" ->
"Forward a Port" -> 8000.  http://localhost:8000 on your PC now reaches
the vLLM server running on m42.
"""
import socket
import sys
import threading

BUFSIZE = 65536


def pump(src, dst):
    """Copy bytes from src to dst until EOF, then half-close dst."""
    try:
        while True:
            data = src.recv(BUFSIZE)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client, target):
    try:
        upstream = socket.create_connection(target)
    except OSError as exc:
        print(f"! connect to {target[0]}:{target[1]} failed: {exc}", flush=True)
        client.close()
        return
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pump, args=(upstream, client), daemon=True).start()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target_host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    listen_port = int(sys.argv[3]) if len(sys.argv) > 3 else port
    target = (target_host, port)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", listen_port))
    server.listen(64)
    print(f"relay: 127.0.0.1:{listen_port} -> {target_host}:{port}", flush=True)

    while True:
        client, addr = server.accept()
        print(f"+ connection from {addr[0]}:{addr[1]}", flush=True)
        threading.Thread(target=handle, args=(client, target), daemon=True).start()


if __name__ == "__main__":
    main()
