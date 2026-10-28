"""多模态能力：图片理解与扫描版 PDF OCR 的统一入口。

所有视觉模型调用都集中在这里，便于替换供应商、限制输入大小和统一错误处理。
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from config import get_settings
from uploads import UploadValidationError, validate_image_bytes

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_PROMPT = (
    "请详细提取并描述这张图片中的内容。如果是图表、表格或公式，"
    "请尽可能输出结构化的 Markdown 文本，保留数字与单位。"
)
_OCR_PROMPT = "请提取这张图片中的所有文本和公式，尽量保持原有的段落结构与阅读顺序。"


class MultimodalError(RuntimeError):
    """视觉模型不可用或调用失败。"""


def _client():
    # 延迟导入 + 延迟初始化，避免缺少依赖或未配置 key 时影响应用启动
    from zai import ZhipuAiClient

    api_key = (os.getenv("ZHIPU_API_KEY") or "").strip()
    if not api_key:
        raise MultimodalError("未配置 ZHIPU_API_KEY,无法进行图片/OCR 识别")
    return ZhipuAiClient(api_key=api_key)


def _call_vlm(data: bytes, prompt: str) -> str:
    settings = get_settings()
    model = settings.vlm_model

    try:
        response = _client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": base64.b64encode(data).decode("ascii")},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            timeout=settings.vlm_timeout,
        )
        content = response.choices[0].message.content
    except MultimodalError:
        raise
    except Exception as exc:  # 供应商 SDK 的异常类型不稳定，统一包装
        logger.exception("视觉模型调用失败(model=%s)", model)
        raise MultimodalError("图片识别服务暂时不可用") from exc

    text = (content or "").strip()
    if not text:
        raise MultimodalError("图片识别结果为空")
    return text


def describe_image_bytes(data: bytes, prompt: str | None = None) -> str:
    settings = get_settings()
    validate_image_bytes(data, settings.max_image_mb * 1024 * 1024)
    return _call_vlm(data, prompt or _DEFAULT_IMAGE_PROMPT)


def describe_image_file(path: str | Path) -> str:
    file_path = Path(path)
    data = file_path.read_bytes()
    return describe_image_bytes(data)


def ocr_pdf_page(page_png: bytes) -> str:
    settings = get_settings()
    validate_image_bytes(page_png, settings.max_image_mb * 1024 * 1024)
    return _call_vlm(page_png, _OCR_PROMPT)


__all__ = [
    "MultimodalError",
    "UploadValidationError",
    "describe_image_bytes",
    "describe_image_file",
    "ocr_pdf_page",
]
