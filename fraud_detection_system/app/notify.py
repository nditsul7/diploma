"""Сповіщення про знайдені BLOCK: e-mail (SMTP) і push (ntfy.sh або власний сервер ntfy).

Налаштування сервера через змінні середовища:
    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_SSL=1 (для порту 465)
    NTFY_SERVER (типово https://ntfy.sh)
Одержувач (e-mail, тема push) задається користувачем у налаштуваннях.
Функції ніколи не кидають виняток: повертають (ok, повідомлення).
"""
from __future__ import annotations

import os
import re
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


def email_configured() -> bool:
    return bool(os.getenv("SMTP_HOST"))


def send_email(to: str, subject: str, body: str) -> tuple[bool, str]:
    if not EMAIL_RE.match(to or ""):
        return False, "Некоректна адреса e-mail"
    host = os.getenv("SMTP_HOST")
    if not host:
        return False, "SMTP не налаштовано (змінна SMTP_HOST)"
    user, password = os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASSWORD", "")
    msg = EmailMessage()
    msg["Subject"], msg["To"] = subject, to
    msg["From"] = os.getenv("SMTP_FROM") or user or "fraud-monitor@localhost"
    msg.set_content(body)
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        if os.getenv("SMTP_SSL") == "1" or port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=10, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port, timeout=10)
        with server:
            if not isinstance(server, smtplib.SMTP_SSL):
                server.starttls(context=ssl.create_default_context())
            if user:
                server.login(user, password)
            server.send_message(msg)
        return True, "Лист надіслано"
    except Exception as exc:  # noqa: BLE001 — сповіщення не повинні ламати перевірку
        return False, f"Не вдалося надіслати e-mail: {exc}"


def send_push(topic: str, title: str, body: str) -> tuple[bool, str]:
    if not TOPIC_RE.match(topic or ""):
        return False, "Тема push: 6–64 символи, латиниця, цифри, _ або -"
    server = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    req = urllib.request.Request(
        f"{server}/{topic}", data=body.encode("utf-8"), method="POST",
        headers={"Title": title.encode("utf-8").decode("latin-1", "ignore") or "Fraud Monitor",
                 "Priority": "high", "Tags": "warning"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True, "Push надіслано"
    except Exception as exc:  # noqa: BLE001
        return False, f"Не вдалося надіслати push: {exc}"


def _summary(label: str, n_block: int, amount: float, rows: int) -> tuple[str, str]:
    subject = f"Fraud Monitor: заблоковано {n_block} транзакцій"
    body = (f"Перевірка «{label}»: {rows} транзакцій.\n"
            f"Знайдено BLOCK: {n_block}\nСума заблокованого: {amount:,.2f}".replace(",", " "))
    return subject, body


def notify_block(cfg: dict, *, label: str, n_block: int, amount: float, rows: int) -> list[str]:
    """Надсилає сповіщення згідно з налаштуваннями користувача. Повертає список повідомлень про результат."""
    n = cfg["notify"]
    if not n["on_block"] or n_block <= 0:
        return []
    subject, body = _summary(label, n_block, amount, rows)
    results = []
    if n["email_enabled"] and n["email"]:
        results.append(send_email(n["email"], subject, body)[1])
    if n["push_enabled"] and n["push_topic"]:
        results.append(send_push(n["push_topic"], "Fraud Monitor: BLOCK", body)[1])
    return results