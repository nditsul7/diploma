"""Авторизація дашборда: користувачі в JSON-файлі, паролі хешуються PBKDF2-HMAC-SHA256.

Керування з командного рядка (з кореня проєкту):
    python -m app.auth add <логін> [--role admin|analyst]
    python -m app.auth list
    python -m app.auth delete <логін>
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

USERS_FILE = Path(os.getenv("USERS_FILE", "data/users.json"))
ITERATIONS = 200_000
ROLES = ("admin", "analyst")
MIN_PASSWORD_LEN = 8
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
DEFAULT_ADMIN = os.getenv("ADMIN_USER", "admin")
DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin12345")
_DUMMY_HASH = "pbkdf2_sha256$200000$" + "00" * 16 + "$" + "00" * 32


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
    except (ValueError, TypeError):
        return False
    return algo == "pbkdf2_sha256" and hmac.compare_digest(digest.hex(), hash_hex)


def load_users() -> dict[str, dict]:
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def save_users(users: dict[str, dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(USERS_FILE, 0o600)
    except OSError:  # Windows
        pass


def _check_credentials(username: str, password: str, role: str | None = None) -> None:
    if not USERNAME_RE.match(username):
        raise ValueError("Логін: 3–32 символи, лише латиниця, цифри та _ . -")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"Пароль має містити щонайменше {MIN_PASSWORD_LEN} символів")
    if role is not None and role not in ROLES:
        raise ValueError(f"Роль має бути однією з: {', '.join(ROLES)}")


def add_user(username: str, password: str, role: str = "analyst") -> None:
    username = username.strip().lower()
    _check_credentials(username, password, role)
    users = load_users()
    if username in users:
        raise ValueError(f"Користувач «{username}» уже існує")
    users[username] = {
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_users(users)


def set_password(username: str, password: str) -> None:
    username = username.strip().lower()
    users = load_users()
    if username not in users:
        raise ValueError("Користувача не знайдено")
    _check_credentials(username, password)
    users[username]["password_hash"] = hash_password(password)
    save_users(users)


def delete_user(username: str) -> None:
    username = username.strip().lower()
    users = load_users()
    if username not in users:
        raise ValueError("Користувача не знайдено")
    admins = [u for u, d in users.items() if d["role"] == "admin"]
    if users[username]["role"] == "admin" and len(admins) == 1:
        raise ValueError("Не можна видалити єдиного адміністратора")
    del users[username]
    save_users(users)


def ensure_default_admin() -> None:
    """Перший запуск: створює адміністратора (логін/пароль з ADMIN_USER/ADMIN_PASSWORD або типові)."""
    if not load_users():
        add_user(DEFAULT_ADMIN, DEFAULT_PASSWORD, "admin")


def uses_default_password(username: str) -> bool:
    user = load_users().get(username.strip().lower())
    return bool(user) and verify_password(DEFAULT_PASSWORD, user["password_hash"])


def authenticate(username: str, password: str) -> dict | None:
    """Повертає {'username', 'role'} або None. Час перевірки не залежить від існування логіна."""
    username = username.strip().lower()
    user = load_users().get(username)
    ok = verify_password(password, user["password_hash"] if user else _DUMMY_HASH)
    return {"username": username, "role": user["role"]} if user and ok else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("username")
    add.add_argument("--role", choices=ROLES, default="analyst")
    sub.add_parser("list")
    sub.add_parser("delete").add_argument("username")
    args = parser.parse_args()

    try:
        if args.cmd == "add":
            pw = getpass.getpass("Пароль: ")
            if pw != getpass.getpass("Повторіть пароль: "):
                raise ValueError("Паролі не збігаються")
            add_user(args.username, pw, args.role)
            print("Користувача створено")
        elif args.cmd == "list":
            for name, d in load_users().items():
                print(f"{name:<20}{d['role']:<10}{d.get('created_at', '')}")
        else:
            delete_user(args.username)
            print("Користувача видалено")
    except ValueError as exc:
        raise SystemExit(f"Помилка: {exc}")


if __name__ == "__main__":
    main()