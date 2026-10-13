#!/bin/bash
# Forward a port from the vLLM compute node to the login node, so that
# VS Code's Ports panel can forward it to your PC (localhost:<PORT>).
#
# Run this ON THE LOGIN NODE (where VS Code is connected). It starts
# TOOLS/port_relay.py in the background: login05:127.0.0.1:<PORT> -> <NODE>:<PORT>
#
# Usage:
#   ./forward_port.sh [PORT] [NODE]
#
#   PORT  port to forward               (default: 8000)
#   NODE  compute node running vLLM     (default: node of your RUNNING vllm_cpu job)
#
# Then in VS Code:
#   1. Command Palette (Ctrl/Cmd+Shift+P) -> "Ports: Focus on Ports View"
#   2. "Forward a Port" -> <PORT>   (VS Code may also auto-detect it)
#   3. Open http://localhost:<PORT> on your PC.
#
# Re-run after every new SLURM job (the job may land on a different node).

set -euo pipefail
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)

PORT=${1:-8000}
NODE=${2:-$(squeue -u "$USER" -n vllm_cpu -h -t RUNNING -o "%N" | head -n 1)}

if [[ -z "${NODE}" ]]; then
    echo "No RUNNING vllm_cpu job found. Start one first, or pass the node: $0 PORT NODE" >&2
    exit 1
fi

PIDFILE="$HOME/.forward_port_${PORT}.pid"
LOGFILE="$HOME/.forward_port_${PORT}.log"

# Stop a previous relay for this port, if any.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")"
    echo "Stopped previous relay (pid $(cat "$PIDFILE"))."
fi

nohup python3 "$TOOLS_DIR/port_relay.py" "$NODE" "$PORT" >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
sleep 1

if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Relay failed to start - see $LOGFILE" >&2
    exit 1
fi

cat <<EOF
Relay started: 127.0.0.1:$PORT (login05) -> $NODE:$PORT   [pid $(cat "$PIDFILE")]
  log:  $LOGFILE
  stop: kill \$(cat $PIDFILE)

Next steps in VS Code:
  1. Command Palette -> "Ports: Focus on Ports View"
  2. "Forward a Port" -> $PORT   (VS Code may also detect it automatically)
  3. Open http://localhost:$PORT on your PC.
EOF
