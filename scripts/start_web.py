#!/usr/bin/env python3
"""测试平台启动入口（跨平台）。

`start.sh`（macOS / Linux）与 `start.bat`（Windows）都只负责「找到解释器 → 建虚拟环境 →
补齐依赖」，真正与操作系统相关的启动动作集中在这里，避免两套脚本各写一遍：

1. 加载项目根目录的 `.env`（已存在的环境变量优先，不会被覆盖，和 `set -a; source .env` 等价）；
2. 释放被占用的监听端口（Windows 走 `taskkill /T`，POSIX 走 lsof / fuser）；
3. 按需在服务可连接后自动打开浏览器；
4. 启动 `web.server`。

用法：
    python scripts/start_web.py [--host 127.0.0.1] [--port 8085] [--open-browser]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WILDCARD_HOSTS = {"", "0.0.0.0", "::", "*"}
# Windows 上 PID 0/4 是内核与 System；端口被它们占住时不该由启动脚本去杀。
WINDOWS_PROTECTED_PIDS = {0, 4}


def force_utf8_stdio() -> None:
    """中文日志不应在非 UTF-8 控制台/重定向下把进程打崩。

    Windows 控制台用不同代码页（cp936 / cp1252 / …）时，`print` 含中文的提示可能抛
    UnicodeEncodeError；这里统一成 UTF-8 + replace，最坏情况只是个别字符显示为占位符。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 已被重定向为不支持 reconfigure 的对象
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def load_dotenv(path: Path) -> list[str]:
    """极简 .env 解析：KEY=VALUE，跳过空行与 # 注释，容忍 export 前缀与引号。

    语义与 `set -a; source .env` 对齐：**已存在的环境变量优先**（bash 侧已导出的值不被覆盖），
    行内 `#` 只有前面是空白时才算注释（与 bash 一致，避免误伤含 # 的密钥）。
    """
    if not path.is_file():
        return []
    loaded: list[str] = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def port_in_use(host: str, port: int, timeout: float = 0.4) -> bool:
    """实际连一次判断端口是否已被监听。

    不用 bind 探测：Windows 的 SO_REUSEADDR 语义与 POSIX 不同，被监听的端口也能 bind 成功，
    会造成「以为空闲 → 启动时才报地址占用」的假象。
    """
    targets = ["127.0.0.1"]
    if host not in WILDCARD_HOSTS and host not in {"127.0.0.1", "localhost"}:
        targets.insert(0, host)
    for target in targets:
        try:
            with socket.create_connection((target, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def parse_windows_listeners(output: str, port: int) -> list[int]:
    """从 `netstat -ano -p TCP` 输出里取出监听指定端口的 PID。

    只认 LISTENING 行（ESTABLISHED 是客户端连接，不是占用方）；IPv6 的 `[::]:8085`
    与 IPv4 的 `0.0.0.0:8085` 都算，PID 0/4 是内核与 System，不参与终止。
    """
    pids: list[int] = []
    for line in output.splitlines():
        parts = line.split()
        # 形如：TCP  0.0.0.0:8085  0.0.0.0:0  LISTENING  1234
        if len(parts) < 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
            continue
        if parts[1].rsplit(":", 1)[-1] != str(port):
            continue
        pid = parts[4]
        if pid.isdigit() and int(pid) not in pids and int(pid) not in WINDOWS_PROTECTED_PIDS:
            pids.append(int(pid))
    return pids


def listening_pids(port: int) -> list[int]:
    """定位监听指定端口的进程；取不到时返回空列表（交由后续 bind 报真实错误）。"""
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=20, errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return []
        return parse_windows_listeners(completed.stdout, port)
    for tool, args in (("lsof", ["lsof", "-tiTCP:%d", "-sTCP:LISTEN"]), ("fuser", ["fuser", "%d/tcp"])):
        if shutil.which(tool) is None:
            continue
        try:
            completed = subprocess.run(
                [arg % port if "%d" in arg else arg for arg in args],
                capture_output=True, text=True, timeout=20, errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            continue
        pids = []
        for token in completed.stdout.split():
            if token.isdigit() and int(token) not in pids:
                pids.append(int(token))
        if pids:
            return pids
    return []


def release_port(host: str, port: int) -> None:
    """端口被占用时终止持有进程，避免每次手动清理。杀不掉时放行，让 bind 报真实原因。"""
    if not port_in_use(host, port):
        return
    pids = [pid for pid in listening_pids(port) if pid != os.getpid()]
    if not pids:
        return
    print(f"端口 {host}:{port} 被进程占用 (PID: {', '.join(str(p) for p in pids)})，正在终止……", flush=True)
    for pid in pids:
        try:
            if os.name == "nt":
                # 只用 terminate() 杀不掉孙进程（pytest-xdist worker 就是这么来的），必须带上 /T。
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, text=True, timeout=20, errors="replace")
            else:
                os.kill(pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
    deadline = time.time() + 5
    while time.time() < deadline and port_in_use(host, port):
        time.sleep(0.2)
    if port_in_use(host, port):
        print(f"警告：{host}:{port} 仍被占用，稍后可能启动失败。", flush=True)


def open_browser_when_ready(url: str, host: str, port: int, timeout: float = 30.0) -> None:
    """等待端口可连接后再开浏览器，避免抢跑打开一个「无法访问」的页面。"""
    def worker() -> None:
        probe_host = "127.0.0.1" if host in WILDCARD_HOSTS else host
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((probe_host, port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        try:
            webbrowser.open(url)
        except Exception:  # 无图形环境 / 无默认浏览器时静默跳过，不影响服务本身
            pass

    threading.Thread(target=worker, name="open-browser", daemon=True).start()


def display_host(host: str) -> str:
    """0.0.0.0 / :: 不是可路由地址：浏览器会把它丢给系统代理并报 502，展示时换成回环地址。"""
    return "127.0.0.1" if host in WILDCARD_HOSTS else host


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description="大模型测试平台启动入口")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址，默认仅本机可访问；需要局域网访问传 0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--open-browser", dest="open_browser", action="store_true",
                        help="服务就绪后自动打开浏览器")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false",
                        help="不自动打开浏览器（用于覆盖 start.bat 的默认行为）")
    parser.set_defaults(open_browser=False)
    args = parser.parse_args(argv)

    loaded = load_dotenv(ROOT / ".env")
    if loaded:
        print(f"已加载配置：.env（{len(loaded)} 项）", flush=True)

    # 子进程（各套件 runner）依赖 PYTHONPATH 找到包；此处统一设置，两个平台的包装脚本都不用再管。
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

    display = display_host(args.host)
    url = f"http://{display}:{args.port}"
    release_port(args.host, args.port)
    if args.open_browser:
        open_browser_when_ready(url, args.host, args.port)

    from web.server import main as serve
    return serve(["--host", args.host, "--port", str(args.port)]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
