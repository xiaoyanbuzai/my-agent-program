"""账号与鉴权：注册、登录、JWT、当前用户依赖。"""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
# from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
import bcrypt

from db import get_db
from models import User

# ---------- 配置 ----------
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------- 密码工具 ----------
# def hash_password(password: str) -> str:
#     return pwd_context.hash(password)


# def verify_password(plain: str, hashed: str) -> bool:
#     return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    # bcrypt 只处理前 72 字节，超出部分截断
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    pw = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pw, hashed.encode("utf-8"))


# ---------- JWT 工具 ----------
def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    """解析 token,返回 user_id。失败抛 JWTError。"""
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return int(payload["sub"])


# ---------- 请求模型 ----------
class AuthRequest(BaseModel):
    username: str
    password: str


# ---------- 依赖：从请求头拿当前用户 ----------
security = HTTPBearer(auto_error=False)

async def get_user_id(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    """FastAPI 依赖：解析 Authorization: Bearer xxx,返回 user_id。"""
    if not credentials or not credentials.scheme.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

# async def get_user_id(authorization: Optional[str] = Header(None)) -> int:
#     """FastAPI 依赖：解析 Authorization: Bearer xxx,返回 user_id。"""
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="未登录",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     token = authorization[7:]
#     try:
#         return decode_token(token)
#     except JWTError:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="token 无效或已过期",
#             headers={"WWW-Authenticate": "Bearer"},
#         )


# ---------- 业务函数 ----------
def register_user(db: Session, username: str, password: str) -> User:
    """注册。用户名已存在则抛 ValueError。"""
    username = username.strip()
    if len(username) < 3:
        raise ValueError("用户名至少 3 个字符")
    if len(password) < 6:
        raise ValueError("密码至少 6 个字符")

    existing = db.query(User).filter_by(username=username).first()
    if existing:
        raise ValueError("用户名已存在")

    user = User(username=username, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_user(db: Session, username: str, password: str) -> User:
    """登录。用户名或密码错误抛 ValueError。"""
    user = db.query(User).filter_by(username=username.strip()).first()
    if not user or not verify_password(password, user.password_hash):
        raise ValueError("用户名或密码错误")
    return user