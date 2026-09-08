#!/usr/bin/env bash
set -Eeuo pipefail

# 一键启动测试平台：bash start.sh
# 可选：bash start.sh --host 0.0.0.0 --port 8085

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：未找到 Python 3，请安装 Python 3.12+，或设置 PYTHON_BIN。" >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "正在创建虚拟环境：$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"

# 检查项目运行时和内置 KVV 官方测试所需的依赖，缺失时自动安装。
if ! "$PYTHON" -c 'import httpx, openai, jsonschema, pytest, xdist, PIL' >/dev/null 2>&1; then
    echo "正在安装/补齐项目依赖……"
    "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
    echo "正在加载配置：.env"
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

HOST="127.0.0.1"
PORT="8085"
SERVER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host=*) HOST="${1#*=}"; shift ;;
        --host) [[ $# -ge 2 ]] || { echo "错误：--host 缺少参数。" >&2; exit 1; }; HOST="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --port) [[ $# -ge 2 ]] || { echo "错误：--port 缺少参数。" >&2; exit 1; }; PORT="$2"; shift 2 ;;
        *) SERVER_ARGS+=("$1"); shift ;;
    esac
done

if "$PYTHON" - "$HOST" "$PORT" <<'PY'
import socket
import sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
try:
    s.bind((host, port))
except OSError:
    raise SystemExit(0)
else:
    s.close()
    raise SystemExit(1)
PY
then
    echo "错误：${HOST}:${PORT} 已被占用，请先停止已有服务或更换端口。" >&2
    exit 1
fi

echo "测试平台启动中：http://${HOST}:${PORT}"
exec "$PYTHON" "$ROOT_DIR/scripts/start_web.py" --host "$HOST" --port "$PORT" "${SERVER_ARGS[@]}"
