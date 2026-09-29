"""确定性图片夹具（纯标准库，无 Pillow / 无外网依赖）。

从 dsv4/core.py 抽出的公共实现，供 DeepSeek / GPT-5.6 等套件的图像识别用例共用：
固定"上红下蓝"纯色图案，保证识别用例稳定、可复现（2026-09-24 立项时确立）。
"""
from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# 图片识别用例使用固定纯色，避免依赖外部 URL、联网下载与随机图案。
IMAGE_TOP_RGB = (220, 20, 20)      # 红
IMAGE_BOTTOM_RGB = (20, 20, 220)   # 蓝


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def png_bands(top_rgb: tuple[int, int, int] = IMAGE_TOP_RGB, bottom_rgb: tuple[int, int, int] = IMAGE_BOTTOM_RGB,
              width: int = 64, height: int = 64) -> bytes:
    """生成上/下两色的 PNG（纯标准库实现，无 Pillow 依赖）。"""
    raw = b""
    for y in range(height):
        rgb = top_rgb if y < height // 2 else bottom_rgb
        raw += b"\x00" + bytes(rgb) * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw, 9)) + _png_chunk(b"IEND", b"")


def png_data_url(**kwargs: Any) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bands(**kwargs)).decode("ascii")


def image_part(url: str, detail: str | None = None) -> dict[str, Any]:
    """Chat Completions 风格的图片内容块。"""
    block: dict[str, Any] = {"url": url}
    if detail:
        block["detail"] = detail
    return {"type": "image_url", "image_url": block}
