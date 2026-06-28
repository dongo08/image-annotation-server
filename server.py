import io
import json
import os
import tempfile
import time
from typing import List, Optional

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from pywhispercpp.model import Model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "whisper-large-v3-q8_0.gguf")

model: Optional[Model] = None

app = FastAPI(title="Whisper ASR + Image Annotation API", version="1.0.0")

# ── 图片标注相关 ──────────────────────────────────────────────

FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑 (Windows)
    "C:/Windows/Fonts/simhei.ttf", # 黑体
    "C:/Windows/Fonts/simsun.ttc", # 宋体
]

EMOJI_FONT_PATH = "C:/Windows/Fonts/seguiemj.ttf"  # Segoe UI Emoji

# Unicode 区间判断是否为 emoji 字符
_EMOJI_RANGES = [
    (0x200D, 0x200D),   # ZWJ
    (0x20E3, 0x20E3),   # 组合按键
    (0x2139, 0x21AA),   # 杂项符号
    (0x2300, 0x27BF),   # 技术符号 / 杂项符号 / 装饰符号
    (0x2934, 0x2935),   # 箭头
    (0x2B05, 0x2B07),   # 箭头
    (0x2B1B, 0x2B1C),   # 方形
    (0x2B50, 0x2B55),   # 星形
    (0x3030, 0x3030),   # 波浪线
    (0x303D, 0x303D),   # 部分标记
    (0x3297, 0x3299),   # 带圈表意文字
    (0xFE00, 0xFE0F),   # 变体选择器
    (0x1F000, 0x1FFFF), # SMP: 表情 / 象形文字 / 旗帜 等
]


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _EMOJI_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _split_by_emoji(text: str) -> list:
    """将文本切分为 (is_emoji, substring) 段"""
    if not text:
        return []
    segments = []
    current = text[0]
    cur_is_emoji = _is_emoji(text[0])
    for ch in text[1:]:
        ch_is_emoji = _is_emoji(ch)
        if ch_is_emoji == cur_is_emoji:
            current += ch
        else:
            segments.append((cur_is_emoji, current))
            current = ch
            cur_is_emoji = ch_is_emoji
    segments.append((cur_is_emoji, current))
    return segments


def _find_font() -> Optional[ImageFont.FreeTypeFont]:
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=20)
    return None


def _find_emoji_font() -> Optional[ImageFont.FreeTypeFont]:
    if os.path.exists(EMOJI_FONT_PATH):
        return ImageFont.truetype(EMOJI_FONT_PATH, size=20)
    return None


class AnnotationBox(BaseModel):
    x: float = Field(..., ge=0, le=1, description="矩形左上角 x，0=左 1=右")
    y: float = Field(..., ge=0, le=1, description="矩形左上角 y，0=上 1=下")
    width: float = Field(..., ge=0, le=1, description="矩形宽度占比")
    height: float = Field(..., ge=0, le=1, description="矩形高度占比")
    text: str = Field(..., description="要插入的文本")


class AnnotationRequest(BaseModel):
    annotations: List[AnnotationBox] = Field(..., description="标注列表")


def _measure_width(text: str, font: ImageFont.FreeTypeFont,
                   emoji_font: Optional[ImageFont.FreeTypeFont],
                   draw: ImageDraw.Draw) -> int:
    """准确测量混合 emoji 的文本像素宽度"""
    if not emoji_font:
        return draw.textbbox((0, 0), text, font=font)[2]
    w = 0
    for is_e, seg in _split_by_emoji(text):
        f = emoji_font if is_e else font
        w += draw.textbbox((0, 0), seg, font=f)[2]
    return w


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, emoji_font: Optional[ImageFont.FreeTypeFont],
               draw: ImageDraw.Draw, max_width: int) -> list:
    """按字符换行，使用混合字体准确测量宽度"""
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        if _measure_width(test, font, emoji_font, draw) > max_width and current:
            lines.append(current)
            current = ch
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [text]


def _draw_mixed_line(draw: ImageDraw.Draw, line: str, x: float, y: float,
                     font: ImageFont.FreeTypeFont, emoji_font: Optional[ImageFont.FreeTypeFont],
                     fill: str = "black"):
    """逐段绘制混合了 emoji 的文本行"""
    if not emoji_font:
        draw.text((x, y), line, fill=fill, font=font)
        return
    segments = _split_by_emoji(line)
    cx = x
    for is_emoji, seg in segments:
        f = emoji_font if is_emoji else font
        draw.text((cx, y), seg, fill=fill, font=f)
        cx += draw.textbbox((0, 0), seg, font=f)[2]


def _draw_annotations(
    img: Image.Image, annotations: List[AnnotationBox]
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    base_font = _find_font()
    emoji_font = _find_emoji_font()
    MIN_FONT = 10
    PAD = 4

    for ann in annotations:
        x0 = int(ann.x * iw)
        y0 = int(ann.y * ih)
        w = int(ann.width * iw)
        h = int(ann.height * ih)

        draw.rectangle([x0, y0, x0 + w, y0 + h], fill="white")

        if not base_font:
            continue

        avail_w = w - PAD * 2
        avail_h = h - PAD * 2

        # 二分查找最大字号
        lo, hi = MIN_FONT, min(h, 200)
        best = None  # (size, lines, line_height)
        while lo <= hi:
            mid = (lo + hi) // 2
            test_font = base_font.font_variant(size=mid)
            test_emoji = emoji_font.font_variant(size=mid) if emoji_font else None
            wrapped = _wrap_text(ann.text, test_font, test_emoji, draw, avail_w)
            # 以 CJK 字体为准计算行高
            line_h = draw.textbbox((0, 0), "Ag", font=test_font)[3]
            if emoji_font:
                line_h = max(line_h, draw.textbbox((0, 0), "Ag", font=test_emoji)[3])
            gap = max(0, len(wrapped) - 1) * 2
            if line_h * len(wrapped) + gap <= avail_h:
                best = (mid, wrapped, line_h)
                lo = mid + 1
            else:
                hi = mid - 1

        if best is None:
            fallback_font = base_font.font_variant(size=MIN_FONT)
            fallback_emoji = emoji_font.font_variant(size=MIN_FONT) if emoji_font else None
            size, wrapped = MIN_FONT, _wrap_text(ann.text, fallback_font, fallback_emoji, draw, avail_w)
            line_h = draw.textbbox((0, 0), "Ag", font=fallback_font)[3]
            if emoji_font:
                line_h = max(line_h, draw.textbbox((0, 0), "Ag", font=fallback_emoji)[3])
        else:
            size, wrapped, line_h = best

        text_font = base_font.font_variant(size=size)
        text_emoji = emoji_font.font_variant(size=size) if emoji_font else None
        gap = max(0, len(wrapped) - 1) * 2
        total_h = line_h * len(wrapped) + gap
        start_y = y0 + (h - total_h) / 2

        for i, line in enumerate(wrapped):
            tw = _measure_width(line, text_font, text_emoji, draw)
            tx = x0 + (w - tw) / 2
            ty = start_y + i * (line_h + 2)
            _draw_mixed_line(draw, line, tx, ty, text_font, text_emoji, fill="black")

    return img


@app.post("/annotate")
async def annotate_image(
    image: UploadFile = File(...),
    annotations: str = Form(..., description="JSON 字符串，格式: [{\"x\":0.1,\"y\":0.1,\"width\":0.3,\"height\":0.1,\"text\":\"hello\"}]"),
):
    try:
        data = json.loads(annotations)
        anns = [AnnotationBox(**item) for item in data]
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"error": "annotations 格式错误，应为包含 x/y/width/height/text 的对象数组"},
            status_code=400,
        )

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = _draw_annotations(img, anns)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def get_model() -> Model:
    global model
    if model is None:
        model = Model(
            model=MODEL_PATH,
            n_threads=4,
            print_progress=False,
            print_realtime=False,
        )
    return model


def format_srt_time(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def format_vtt_time(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(seg.t0)} --> {format_srt_time(seg.t1)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{format_vtt_time(seg.t0)} --> {format_vtt_time(seg.t1)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_json(segments) -> list:
    return [
        {
            "id": i,
            "start_ms": seg.t0,
            "end_ms": seg.t1,
            "start": format_srt_time(seg.t0).replace(",", "."),
            "end": format_srt_time(seg.t1).replace(",", "."),
            "text": seg.text.strip(),
        }
        for i, seg in enumerate(segments, 1)
    ]


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    fmt: str = Query(default="json", pattern="^(json|srt|vtt|txt)$"),
    language: str = Query(default="zh", description="语言代码，如 zh/en/ja/auto"),
    translate: bool = Query(default=False, description="翻译为英文（仅非英语有效）"),
    initial_prompt: str = Query(default="", description="初始提示文本"),
):
    ext = os.path.splitext(file.filename or "audio.wav")[-1]
    m = get_model()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        segments = m.transcribe(
            tmp_path,
            language=language if language != "auto" else "",
            detect_language=(language == "auto"),
            translate=translate,
            initial_prompt=initial_prompt if initial_prompt else None,
            no_timestamps=False,
        )
        elapsed = time.time() - t0

        if fmt == "srt":
            content = segments_to_srt(segments)
            return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
        elif fmt == "vtt":
            content = segments_to_vtt(segments)
            return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
        elif fmt == "txt":
            text = "".join(seg.text for seg in segments)
            return PlainTextResponse(text, media_type="text/plain; charset=utf-8")
        else:
            return {
                "format": fmt,
                "language": language,
                "elapsed_s": round(elapsed, 2),
                "segments": segments_to_json(segments),
                "text": "".join(seg.text for seg in segments),
            }
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    m = get_model()
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "languages": m.available_languages()[:10],
    }


if __name__ == "__main__":
    import socket
    import uvicorn

    PORT = 8700

    # 显式创建双栈 socket：同时支持 IPv4 + IPv6
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)  # 双栈模式
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", PORT))

    print(f"服务器已启动 → http://0.0.0.0:{PORT}  (IPv4)")
    print(f"              → http://[::]:{PORT}        (IPv6)")

    config = uvicorn.Config(app, host=None, port=None, log_level="info")
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
