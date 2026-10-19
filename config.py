"""集中配置：环境变量、目录、模型与检索参数。

所有模块都通过 ``get_settings()`` 读取配置，避免散落的 ``os.getenv`` 和路径常量。
环境变量示例见 ``.env.example``。
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 必须在任何 huggingface_hub / sentence-transformers 导入之前生效：
# 这些库在导入时读取一次 HF_HUB_OFFLINE，之后再设置环境变量不会改变行为。
if os.getenv("HF_HUB_OFFLINE", "1").strip().lower() not in {"0", "false", "no", "off"}:
    os.environ["HF_HUB_OFFLINE"] = "1"


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"环境变量 {name} 必须是整数，当前为 {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"环境变量 {name} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"环境变量 {name} 不能大于 {maximum}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"环境变量 {name} 必须是数字，当前为 {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_exts(name: str, default: str) -> frozenset[str]:
    raw = _env_str(name, default)
    exts = {
        item.strip().lower().lstrip(".")
        for item in raw.split(",")
        if item.strip()
    }
    if not exts:
        raise RuntimeError(f"环境变量 {name} 不能为空")
    return frozenset(exts)


def _default_query_prefix(model: str) -> str:
    """BGE 中文 v1.5 系列建议给短查询加指令前缀。"""
    if "bge" in model.lower():
        return "为这个句子生成表示以用于检索相关文章："
    return ""


def _persistent_dev_secret(data_dir: Path) -> str:
    """开发环境没有 JWT_SECRET 时，生成并复用本地密钥，避免重启后 token 全部失效。"""
    secret_file = data_dir / ".jwt_secret"
    try:
        if secret_file.is_file():
            existing = secret_file.read_text(encoding="utf-8").strip()
            if len(existing) >= 32:
                return existing
        data_dir.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(48)
        secret_file.write_text(generated, encoding="utf-8")
        try:
            secret_file.chmod(0o600)
        except OSError:
            pass
        logger.warning(
            "JWT_SECRET 未配置，已生成开发密钥并保存到 %s。"
            "生产环境请通过环境变量配置 JWT_SECRET。",
            secret_file,
        )
        return generated
    except OSError:
        logger.warning("无法持久化开发密钥，本次使用进程级临时密钥，重启后 token 会失效。")
        return secrets.token_urlsafe(48)


@dataclass(frozen=True)
class Settings:
    # 运行环境
    app_env: str
    debug: bool

    # 目录
    base_dir: Path
    data_dir: Path
    knowledge_dir: Path
    chroma_dir: Path
    templates_dir: Path
    log_dir: Path

    # 安全
    secret_key: str
    token_expire_days: int
    max_upload_mb: int
    max_image_mb: int
    max_pdf_mb: int
    max_question_chars: int
    allowed_upload_exts: frozenset[str]
    allowed_image_exts: frozenset[str]
    fetch_allowed_hosts: frozenset[str]

    # 数据库
    database_url: str

    # 模型
    llm_model: str
    llm_timeout: float
    vlm_model: str
    vlm_timeout: float
    embedding_model: str
    embedding_fallback_model: str
    embedding_query_prefix: str
    reranker_model: str
    hf_offline: bool
    model_cache_dir: str

    # 切块
    parent_chunk_size: int
    parent_chunk_overlap: int
    max_parent_chars: int
    child_chunk_size: int
    child_chunk_overlap: int
    min_child_chars: int

    # 检索
    retrieve_candidates: int
    retrieve_top_k: int
    rerank_min_score: float
    hybrid_weight_vector: float
    bm25_max_docs: int
    memory_limit: int
    recursion_limit: int

    # 抓取
    fetch_timeout: float
    fetch_max_chars: int
    fetch_max_mb: int
    ocr_max_pages: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取一次环境变量并缓存；测试可用 ``get_settings.cache_clear()`` 重置。"""
    app_env = _env_str("APP_ENV", "development").lower()

    base_dir = BASE_DIR
    data_dir = Path(_env_str("DATA_DIR", str(base_dir / "data"))).resolve()
    knowledge_dir = Path(
        _env_str("KNOWLEDGE_DIR", str(base_dir / "knowledge"))
    ).resolve()

    secret = _env_str("JWT_SECRET")
    if not secret:
        if app_env in {"production", "prod"}:
            raise RuntimeError(
                "生产环境必须配置 JWT_SECRET（例如 `python -c \"import secrets;print(secrets.token_urlsafe(48))\"`）"
            )
        secret = _persistent_dev_secret(data_dir)
    elif len(secret) < 32:
        raise RuntimeError("JWT_SECRET 长度至少 32 个字符")

    allowed_upload = _env_exts(
        "ALLOWED_UPLOAD_EXTS",
        "pdf,docx,md,markdown,txt,png,jpg,jpeg,webp",
    )
    allowed_images = frozenset({"png", "jpg", "jpeg", "webp"})
    if not allowed_images <= allowed_upload:
        raise RuntimeError("ALLOWED_UPLOAD_EXTS 必须包含 png/jpg/jpeg/webp")

    embedding_model = _env_str("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    embedding_prefix = _env_str(
        "EMBEDDING_QUERY_PREFIX", _default_query_prefix(embedding_model)
    )

    vector_weight = _env_float("HYBRID_VECTOR_WEIGHT", 0.5)
    if not 0.0 <= vector_weight <= 1.0:
        raise RuntimeError("HYBRID_VECTOR_WEIGHT 必须在 0~1 之间")

    fetch_hosts_raw = _env_str("FETCH_ALLOWED_HOSTS", "")
    fetch_hosts = frozenset(
        h.strip().lower() for h in fetch_hosts_raw.split(",") if h.strip()
    )

    return Settings(
        app_env=app_env,
        debug=_env_bool("APP_DEBUG", False),
        base_dir=base_dir,
        data_dir=data_dir,
        knowledge_dir=knowledge_dir,
        chroma_dir=data_dir / "chroma_db",
        templates_dir=base_dir / "templates",
        log_dir=data_dir / "logs",
        secret_key=secret,
        token_expire_days=_env_int("TOKEN_EXPIRE_DAYS", 7, minimum=1, maximum=90),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 30, minimum=1, maximum=500),
        max_image_mb=_env_int("MAX_IMAGE_MB", 10, minimum=1, maximum=100),
        max_pdf_mb=_env_int("MAX_PDF_MB", 50, minimum=1, maximum=500),
        max_question_chars=_env_int("MAX_QUESTION_CHARS", 8000, minimum=100),
        allowed_upload_exts=allowed_upload,
        allowed_image_exts=allowed_images,
        fetch_allowed_hosts=fetch_hosts,
        database_url=_env_str(
            "DATABASE_URL",
            f"sqlite:///{(data_dir / 'app.db').as_posix()}",
        ),
        llm_model=_env_str("LLM_MODEL", "deepseek-flash"),
        llm_timeout=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        vlm_model=_env_str("VLM_MODEL", "glm-4.6v-flash"),
        vlm_timeout=_env_float("VLM_TIMEOUT_SECONDS", 60.0),
        embedding_model=embedding_model,
        embedding_fallback_model=_env_str(
            "EMBEDDING_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_query_prefix=embedding_prefix,
        reranker_model=_env_str(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        hf_offline=_env_bool("HF_HUB_OFFLINE", True),
        model_cache_dir=_env_str("MODEL_CACHE_DIR", ""),
        parent_chunk_size=_env_int("PARENT_CHUNK_SIZE", 3000, minimum=500),
        parent_chunk_overlap=_env_int("PARENT_CHUNK_OVERLAP", 200, minimum=0),
        max_parent_chars=_env_int("MAX_PARENT_CHARS", 1500, minimum=300),
        child_chunk_size=_env_int("CHILD_CHUNK_SIZE", 400, minimum=100),
        child_chunk_overlap=_env_int("CHILD_CHUNK_OVERLAP", 50, minimum=0),
        min_child_chars=_env_int("MIN_CHILD_CHARS", 30, minimum=1),
        retrieve_candidates=_env_int("RETRIEVE_CANDIDATES", 20, minimum=5, maximum=100),
        retrieve_top_k=_env_int("RETRIEVE_TOP_K", 4, minimum=1, maximum=20),
        rerank_min_score=_env_float("RERANK_MIN_SCORE", -10.0),
        hybrid_weight_vector=vector_weight,
        bm25_max_docs=_env_int("BM25_MAX_DOCS", 20000, minimum=100),
        memory_limit=_env_int("MEMORY_LIMIT", 3, minimum=0, maximum=20),
        recursion_limit=_env_int("AGENT_RECURSION_LIMIT", 24, minimum=2, maximum=60),
        fetch_timeout=_env_float("FETCH_TIMEOUT_SECONDS", 20.0),
        fetch_max_chars=_env_int("FETCH_MAX_CHARS", 8000, minimum=1000),
        fetch_max_mb=_env_int("FETCH_MAX_MB", 2, minimum=1, maximum=50),
        ocr_max_pages=_env_int("OCR_MAX_PAGES", 20, minimum=1, maximum=200),
    )


def ensure_directories() -> None:
    """创建运行时目录。"""
    settings = get_settings()
    for path in (
        settings.data_dir,
        settings.chroma_dir,
        settings.knowledge_dir,
        settings.log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def user_dir(kind: str, user_id: int) -> Path:
    """返回某租户在 knowledge/<kind>/ 下的目录，并保证 user_id 合法。"""
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("user_id 必须是正整数")
    if kind not in {"documents", "web", "tmp"}:
        raise ValueError(f"未知目录类型：{kind}")
    path = get_settings().knowledge_dir / kind / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_hf_environment() -> None:
    """统一设置 Hugging Face 离线环境变量。"""
    settings = get_settings()
    if settings.hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if settings.model_cache_dir:
        os.environ.setdefault("HF_HOME", settings.model_cache_dir)
