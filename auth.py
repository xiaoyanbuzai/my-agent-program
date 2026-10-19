"""账号与鉴权：bcrypt 密码、JWT、当前用户依赖。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import get_settings
from models import User

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
BCRYPT_MAX_BYTES = 72
MIN_PASSWORD_BYTES = 8
MAX_USERNAME_LENGTH = 64


class AuthRequest(BaseModel):
    username: str
    password: str


def _secret_key() -> str:
    return get_settings().secret_key


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_BYTES:
        raise ValueError("密码过长（bcrypt 上限 72 字节）")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    encoded = plain.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode("utf-8"))
    except ValueError:
        logger.warning("密码哈希格式异常")
        return False


def create_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(days=get_settings().token_expire_days),
    }
    return jwt.encode(payload, _secret_key(), algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    """解析 token 返回 user_id；失败抛 ``JWTError`` / ``ValueError``。"""
    payload = jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
    subject = payload.get("sub")
    if subject is None:
        raise JWTError("token 缺少 sub")
    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise JWTError("token 中的用户标识无效") from exc
    if user_id <= 0:
        raise JWTError("token 中的用户标识无效")
    return user_id


security = HTTPBearer(auto_error=False)


async def get_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def register_user(db: Session, username: str, password: str) -> User:
    username = (username or "").strip()
    if len(username) < 3 or len(username) > MAX_USERNAME_LENGTH:
        raise ValueError("用户名长度需为 3~64 个字符")
    if any(char.isspace() for char in username):
        raise ValueError("用户名不能包含空格")

    password_bytes = (password or "").encode("utf-8")
    if len(password_bytes) < MIN_PASSWORD_BYTES:
        raise ValueError("密码至少 8 个字符")
    if len(password_bytes) > BCRYPT_MAX_BYTES:
        raise ValueError("密码过长（最多 72 字节）")

    if db.query(User).filter_by(username=username).first():
        raise ValueError("用户名已存在")

    user = User(username=username, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_user(db: Session, username: str, password: str) -> User:
    user = db.query(User).filter_by(username=(username or "").strip()).first()
    if not user or not verify_password(password or "", user.password_hash):
        raise ValueError("用户名或密码错误")
    return user
