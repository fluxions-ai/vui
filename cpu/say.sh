#!/usr/bin/env bash
# Warm streaming vui TTS on CPU. RTF>1, prod-normalized, 50ms end-crossfade.
#
#   say.sh --server         boot the resident daemon (loads 3.6GB once; blocks, serving until killed)
#   say.sh "hello there"    speak it (auto-boots the daemon in the background if not already up)
#   say.sh --quit           stop the daemon
#
# Env: VOICE=maeve|abraham|... TEMP=0.6   (restart the daemon to change voice)
cd "$(dirname "$0")"
SOCK="${VUI_SOCK:-/tmp/vui.sock}"

case "$1" in
  --server)
    echo "booting vui daemon (one-time load ~5s)..." >&2
    exec .venv/bin/python vui_daemon.py        # foreground; serves until Ctrl-C / killed
    ;;
  --quit)
    if [ -S "$SOCK" ]; then
      printf '__QUIT__' | .venv/bin/python -c "import socket,sys;s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect('$SOCK');s.sendall(sys.stdin.read().encode())" 2>/dev/null || true
      echo "daemon stopped"
    else
      echo "no daemon running"
    fi
    exit 0
    ;;
esac

# speak: auto-boot the daemon in the background if its socket isn't up yet
if [ ! -S "$SOCK" ]; then
  echo "starting vui daemon (one-time load ~5s)..." >&2
  nohup .venv/bin/python vui_daemon.py > /tmp/vui_daemon.log 2>&1 &
  for i in $(seq 1 80); do [ -S "$SOCK" ] && break; sleep 0.25; done
  [ -S "$SOCK" ] || { echo "daemon failed to start; see /tmp/vui_daemon.log" >&2; exit 1; }
fi

printf '%s' "$*" | .venv/bin/python -c "
import socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect('$SOCK')
s.sendall(sys.stdin.read().encode())
print(s.recv(4096).decode())"
