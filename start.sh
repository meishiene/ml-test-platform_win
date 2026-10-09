#!/usr/bin/env bash
set -Eeuo pipefail

# 一键启动测试平台：bash start.sh
# 可选：bash start.sh --host 0.0.0.0 --port 8085 --open-browser
#
# 本脚本只负责「找到解释器 → 建虚拟环境 → 补齐依赖」这三件与 shell 相关的事；
# .env 加载、端口占用释放、浏览器打开、服务启动都在 scripts/start_web.py 里，
# 与 Windows 的 start.bat 共用同一份实现（改启动逻辑只改那一处）。

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

# 监听地址：默认 0.0.0.0（保持平台原有行为）；随后的 --host/--port 由入口脚本解析，可覆盖默认值。
exec "$PYTHON" "$ROOT_DIR/scripts/start_web.py" \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8085}" "$@"
