# -*- coding: utf-8 -*-
"""
ingestion/auth_login.py — 电网数据平台自动登录，获取 API Token
================================================================
Token 有效期 ~5 天，过期后自动调用登录接口刷新，写入 grid_token.txt。

使用：
  python -m ingestion.auth_login                # 登录并更新 token
  python -m ingestion.auth_login --dry-run      # 仅打印 token，不写文件
"""
import sys
import io
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

TOKEN_FILE = PROJECT_ROOT / "grid_token.txt"
LOGIN_URL = settings.grid_login_url or \
    "https://lingfeng-saas.tradingthink.cn/api/auth/login/v3"


def login(dry_run: bool = False) -> str:
    """调用登录接口获取 token，写入 grid_token.txt。

    Returns
    -------
    str : 新的 JWT token
    """
    username = settings.grid_api_username
    password = settings.grid_api_password

    if not username or not password:
        raise ValueError(
            "未配置登录凭证。请在 .env 中设置:\n"
            "  GRID_API_USERNAME=你的加密用户名\n"
            "  GRID_API_PASSWORD=你的加密密码"
        )

    payload = {
        "code": None,          # 验证码，无需
        "password": password,
        "username": username,
        "platform": settings.grid_api_platform,
    }
    headers = {"Content-Type": "application/json"}

    print(f"  [Login] POST {LOGIN_URL}")
    print(f"  [Login] username={username[:20]}...")

    try:
        import urllib.request
        req = urllib.request.Request(
            LOGIN_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"登录 HTTP 错误 {e.code}: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"登录请求失败: {e}")

    if raw.get('code') != 200:
        raise RuntimeError(
            f"登录失败: code={raw.get('code')}, message={raw.get('message', 'N/A')}"
        )

    data = raw['data']
    token = data.get('token', '')
    expire_time = data.get('expireTime', 'unknown')
    user_name = data.get('sysUser', {}).get('userRealName', 'unknown')

    if not token:
        raise RuntimeError("登录响应中未找到 token")

    print(f"  [Login] 成功 — 用户: {user_name}")
    print(f"  [Login] 过期时间: {expire_time}")
    print(f"  [Login] Token 长度: {len(token)}")

    if not dry_run:
        TOKEN_FILE.write_text(token, encoding='utf-8')
        print(f"  [Login] Token 已写入 {TOKEN_FILE}")
    else:
        print(f"  [Login] DRY-RUN — Token 未写入文件")

    return token


def get_valid_token() -> str:
    """获取有效 token：优先读本地文件，过期则自动登录刷新。

    Returns
    -------
    str : 有效的 JWT token
    """
    # 1. 尝试读本地 token 文件
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding='utf-8').strip()
        if token:
            return token

    # 2. 回退到 .env
    if settings.grid_api_token:
        return settings.grid_api_token

    # 3. 自动登录
    print("  [Auth] Token 不存在，自动登录...")
    return login()


def main():
    parser = argparse.ArgumentParser(description="电网数据平台自动登录")
    parser.add_argument('--dry-run', action='store_true', help='仅打印 token，不写文件')
    args = parser.parse_args()

    print("=" * 60)
    print("  Auth Login — 电网平台 Token 获取")
    print("=" * 60)

    try:
        token = login(dry_run=args.dry_run)
        print(f"\n  Token: {token[:60]}...")
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
