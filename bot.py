# -*- coding: utf-8 -*-
"""
Личный Telegram-бот на базе Z.ai API (модели GLM).

Возможности:
  - Диалог с историей (сохраняется в SQLite, переживает перезапуск)
  - Стриминг ответа «по буквам» — ответ печатается в сообщении в реальном времени
  - Переключение моделей: команда /model или кнопка «🧠 Модель»
  - Панель быстрых подсказок: /prompts или кнопка «💡 Подсказки»
    (готовые запросы + шаблоны «исправить/перевести/объяснить текст»)
  - Распознавание фотографий (бесплатная vision-модель GLM-4.6V-Flash)
  - Белый список пользователей (только вы и ваша жена)
  - Длинные ответы автоматически режутся под лимит Telegram (4096 симв.)

Запуск: см. README.md
"""

import asyncio
import base64
import io
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    ForceReply,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.utils.chat_action import ChatActionSender
from aiogram.utils.keyboard import InlineKeyboardBuilder

# ----------------------------------------------------------------------------
# Конфигурация (читается из .env)
# ----------------------------------------------------------------------------

load_dotenv()

# Токен: основное имя — TELEGRAM_BOT_TOKEN. Хостинг Bothost подставляет его
# автоматически при создании бота; там же есть дубли BOT_TOKEN и API_TOKEN.
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
    or os.getenv("API_TOKEN")
    or ""
)
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "glm-4.7-flash")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты — дружелюбный и полезный ассистент в личном Telegram-боте. "
    "Отвечай на языке собеседника, по делу и без лишней воды, "
    "если не попросили иначе.",
)
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", "20"))  # сколько последних сообщений отправлять модели
MAX_HISTORY_ROWS = int(os.getenv("MAX_HISTORY_ROWS", "200"))  # сколько сообщений хранить в БД на пользователя
MAX_PHOTO_MB = int(os.getenv("MAX_PHOTO_MB", "10"))

# Белый список Telegram ID через запятую. Пусто = разрешено всем (не рекомендуется).
ALLOWED_USER_IDS = {
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x
}

# Доступные модели: id -> (подпись для кнопки, поддерживает картинки)
MODELS = {
    "glm-4.7-flash": ("GLM-4.7-Flash — бесплатная 🆓", False),
    "glm-4.6v-flash": ("GLM-4.6V-Flash — бесплатная, зрение 👁", True),
    "glm-5.3-flash": ("GLM-5.3-Flash — ~$0.0003/сообщение ⚡", False),
    "glm-5.3-flashx": ("GLM-5.3-FlashX — быстрее и умнее 💪", False),
    "glm-5.3": ("GLM-5.3 — флагман 🏆", False),
}

VISION_MODEL = "glm-4.6v-flash"  # всегда используем её для фото (бесплатно)

# Стриминг: как часто обновлять сообщение (Telegram не любит частые правки)
EDIT_INTERVAL = 1.1  # секунды между правками сообщения
STREAM_SEG_LIMIT = 3900  # длина сегмента одного сообщения (лимит TG — 4096)
TG_LIMIT = 4096
CURSOR = "▌"  # «печатающий» курсор в конце стрима

# ----------------------------------------------------------------------------
# Быстрые подсказки
#   prompt  — готовый запрос (отправляется в модель сразу по нажатию)
#   tpl     — шаблон с {text}: бот попросит прислать текст и подставит его
# ----------------------------------------------------------------------------

PROMPTS = {
    "dinner": {"label": "🍳 Ужин за 30 минут",
               "prompt": "Предложи 5 простых рецептов ужина, которые готовятся не дольше 30 минут. Для каждого — краткие шаги и список продуктов."},
    "date": {"label": "💡 Идея для вечера вдвоём",
             "prompt": "Предложи 5 идей романтического вечера дома: атмосферно, без больших затрат и подготовки."},
    "weekend": {"label": "📅 План на выходные",
                "prompt": "Составь интересный план на выходные для двоих: суббота и воскресенье, с примерным расписанием и альтернативами на случай плохой погоды."},
    "movies": {"label": "🎬 Что посмотреть",
               "prompt": "Посоветуй 5 хороших фильмов для семейного вечера. На каждый — одна-две фразы, о чём он и почему стоит посмотреть."},
    "joke": {"label": "😄 Анекдот",
             "prompt": "Расскажи свежий смешной анекдот."},
    "advice": {"label": "🧘 Как отдохнуть после работы",
               "prompt": "Дай 5 простых и действенных способов восстановиться после напряжённого рабочего дня."},
    "fix_text": {"label": "✍️ Исправить текст",
                 "tpl": "Исправь грамматику, пунктуацию и стиль. Верни только исправленный текст без комментариев:\n\n{text}"},
    "translate": {"label": "🌍 Перевести текст",
                  "tpl": "Переведи текст. Если он на русском — переведи на английский, иначе — на русский. Верни только перевод:\n\n{text}"},
    "explain": {"label": "🧠 Объяснить простыми",
                "tpl": "Объясни следующий текст или термин простыми словами, как другу, с бытовым примером:\n\n{text}"},
}

# Кнопки постоянной клавиатуры (панель под внизу экрана)
BTN_PROMPTS = "💡 Подсказки"
BTN_MODEL = "🧠 Модель"
BTN_RESET = "🗑 Новый диалог"

# Отложенный ввод для шаблонных подсказок: user_id -> pid. Живёт в памяти.
PENDING: dict[int, str] = {}

# ----------------------------------------------------------------------------
# Логирование
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("glm-bot")
logging.getLogger("aiogram.event").setLevel(logging.WARNING)

# ----------------------------------------------------------------------------
# База данных (SQLite)
# Amvera: задайте DATA_DIR=/data. Bothost: каталог /app/data — определяется
# автоматически (можно и явно: DATA_DIR=/app/data). База переживает перезапуски.
# Локально база лежит рядом со скриптом (bot_data.db).
# ----------------------------------------------------------------------------

_data_dir = os.getenv("DATA_DIR", "")
if not _data_dir and Path("/app/data").is_dir():  # Bothost: постоянное хранилище
    _data_dir = "/app/data"
if _data_dir:
    Path(_data_dir).mkdir(parents=True, exist_ok=True)  # если папки вдруг нет — создадим
DB_PATH = (Path(_data_dir) / "bot_data.db") if _data_dir else Path(__file__).with_name("bot_data.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                   id         INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_id    INTEGER NOT NULL,
                   role       TEXT NOT NULL,
                   content    TEXT NOT NULL,
                   created_at TEXT DEFAULT (datetime('now'))
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS settings (
                   user_id INTEGER PRIMARY KEY,
                   model   TEXT NOT NULL
               )"""
        )


def add_message(user_id: int, role: str, content) -> None:
    """content — str или list (для фото); храним как JSON-строку."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, json.dumps(content, ensure_ascii=False)),
        )
        # держим БД компактной: оставляем только последние MAX_HISTORY_ROWS сообщений
        conn.execute(
            """DELETE FROM messages WHERE user_id = ? AND id NOT IN (
                   SELECT id FROM messages WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?
               )""",
            (user_id, user_id, MAX_HISTORY_ROWS),
        )


def get_history(user_id: int, limit: int = CONTEXT_MESSAGES):
    """Возвращает последние `limit` сообщений в хронологическом порядке:
    список кортежей (role, content), где content уже распакован из JSON."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [(role, json.loads(raw)) for role, raw in reversed(rows)]


def reset_history(user_id: int) -> int:
    with _db() as conn:
        cur = conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        return cur.rowcount


def history_count(user_id: int) -> int:
    with _db() as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
        ).fetchone()
        return n


def get_user_model(user_id: int) -> str:
    with _db() as conn:
        row = conn.execute(
            "SELECT model FROM settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row and row[0] in MODELS:
        return row[0]
    return DEFAULT_MODEL if DEFAULT_MODEL in MODELS else "glm-4.7-flash"


def set_user_model(user_id: int, model_id: str) -> None:
    with _db() as conn:
        conn.execute(
            """INSERT INTO settings (user_id, model) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET model = excluded.model""",
            (user_id, model_id),
        )


# ----------------------------------------------------------------------------
# Клиент Z.ai (OpenAI-совместимый API)
# ----------------------------------------------------------------------------

client = AsyncOpenAI(api_key=ZAI_API_KEY, base_url=ZAI_BASE_URL)


def build_api_messages(user_id: int, model_id: str):
    """Собирает историю в формат API. Если модель не vision — старые фото
    заменяются текстовой пометкой, чтобы не тратить токены."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    supports_vision = MODELS.get(model_id, (None, False))[1]
    for role, content in get_history(user_id):
        if isinstance(content, list):  # сообщение с фото
            if supports_vision:
                msgs.append({"role": role, "content": content})
            else:
                text_part = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                ).strip()
                msgs.append({"role": role, "content": f"[фото] {text_part}".strip()})
        else:
            msgs.append({"role": role, "content": content})
    return msgs


async def ask_zai(model_id: str, api_messages: list) -> str:
    """Обычный (нестримовый) запрос — используется для фото."""
    resp = await client.chat.completions.create(
        model=model_id, messages=api_messages
    )
    return resp.choices[0].message.content or "…"


async def ask_zai_stream(model_id: str, api_messages: list):
    """Асинхронный генератор кусочков текста из стримового ответа Z.ai."""
    stream = await client.chat.completions.create(
        model=model_id, messages=api_messages, stream=True
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ----------------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------------

router = Router()


class ModelCB(CallbackData, prefix="model"):
    model_id: str


class PromptCB(CallbackData, prefix="p"):
    pid: str


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


def split_text(text: str, limit: int = 4000) -> list[str]:
    """Режет длинный текст на части < 4096 символов (лимит Telegram)."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


def model_keyboard(current: str):
    kb = InlineKeyboardBuilder()
    for model_id, (label, _) in MODELS.items():
        mark = "✅ " if model_id == current else ""
        kb.button(text=mark + label, callback_data=ModelCB(model_id=model_id).pack())
    kb.adjust(1)  # кнопки в один столбец
    return kb.as_markup()


def prompts_keyboard():
    kb = InlineKeyboardBuilder()
    for pid, p in PROMPTS.items():
        kb.button(text=p["label"], callback_data=PromptCB(pid=pid).pack())
    kb.button(text="✖️ Закрыть", callback_data=PromptCB(pid="-").pack())
    kb.adjust(2)  # по две кнопки в ряд
    return kb.as_markup()


def main_reply_kb() -> ReplyKeyboardMarkup:
    """Постоянная клавиатура внизу экрана — чтобы жене не пришлось
    набирать слэш-команды."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_PROMPTS), KeyboardButton(text=BTN_MODEL)],
            [KeyboardButton(text=BTN_RESET)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Напишите сообщение…",
    )


async def _safe_edit(msg: Message, text: str, final: bool = False) -> None:
    """edit_text с защитой от рейт-лимитов Telegram и «message is not modified»."""
    content = text if final else text + CURSOR
    try:
        await msg.edit_text(content)
    except TelegramRetryAfter as e:
        await asyncio.sleep(e.retry_after + 0.2)
        try:
            await msg.edit_text(content)
        except Exception:
            pass
    except TelegramBadRequest:
        pass  # чаще всего «message is not modified» — это нормально


async def stream_answer(anchor: Message, model_id: str, api_messages: list) -> str:
    """Стримит ответ Z.ai в чат: сообщение периодически редактируется,
    создаётся эффект живой печати. Длинные ответы автоматически
    продолжаются следующими сообщениями. Возвращает полный текст."""
    full = ""
    seg_from = 0        # индекс начала текущего сегмента в full
    msg: Message | None = None
    shown = ""          # что уже показано в текущем сегменте (без курсора)
    last_edit = 0.0
    t0 = time.monotonic()

    async for delta in ask_zai_stream(model_id, api_messages):
        full += delta
        cur = full[seg_from:]
        if msg is None:
            # первое сообщение отправляем быстро, но не с одного символа
            if len(cur.strip()) >= 2 or time.monotonic() - t0 > 3:
                msg = await anchor.answer(CURSOR)
                last_edit = time.monotonic()
            else:
                continue
        now = time.monotonic()
        if len(cur) >= STREAM_SEG_LIMIT:
            # сегмент заполнен — фиксируем его и открываем новое сообщение
            await _safe_edit(msg, cur[:STREAM_SEG_LIMIT], final=True)
            seg_from += STREAM_SEG_LIMIT
            msg = await anchor.answer(CURSOR)
            shown = ""
            last_edit = time.monotonic()
        elif now - last_edit >= EDIT_INTERVAL and cur != shown:
            await _safe_edit(msg, cur)
            shown = cur
            last_edit = now

    if msg is None:
        # модель вернула пустой/крохотный стрим — отправим как есть
        await anchor.answer(full or "…")
        return full

    rest = full[seg_from:]
    if rest and rest != shown:
        if len(rest) <= TG_LIMIT:
            await _safe_edit(msg, rest, final=True)  # финал: снимаем курсор
        else:
            await _safe_edit(msg, rest[:STREAM_SEG_LIMIT], final=True)
            for part in split_text(rest[STREAM_SEG_LIMIT:], TG_LIMIT - 10):
                await anchor.answer(part)
    elif shown:
        await _safe_edit(msg, shown, final=True)  # снять курсор
    else:
        # в сообщение ни разу ничего не дописали — убираем висящий курсор
        try:
            await msg.delete()
        except Exception:
            await _safe_edit(msg, "…", final=True)
    return full


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "⏳ Слишком много запросов к API. Подождите немного и попробуйте снова."
    if isinstance(exc, APIConnectionError):
        return "🌐 Не удалось связаться с сервером Z.ai. Проверьте интернет и повторите."
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", "?")
        log.error("API error %s: %s", code, exc)
        if code == 401:
            return "🔑 API отклонил ключ (401). Проверьте ZAI_API_KEY в .env"
        if code == 404:
            return "🤷 Модель не найдена (404). Проверьте ID модели на docs.z.ai"
        return f"⚠️ Ошибка API (код {code}). Попробуйте ещё раз."
    log.exception("Неизвестная ошибка")
    return "⚠️ Что-то пошло не так. Подробности в логах."


async def run_prompt(anchor: Message, uid: int, model_id: str, prompt: str) -> None:
    """Сохраняет запрос в историю, стримит ответ, сохраняет ответ."""
    add_message(uid, "user", prompt)
    try:
        answer = await stream_answer(anchor, model_id, build_api_messages(uid, model_id))
    except Exception as exc:
        # убираем «осиротевший» вопрос, чтобы он не попал в контекст дважды
        reset_only_last_user_message(uid)
        await anchor.answer(friendly_error(exc))
        return
    add_message(uid, "assistant", answer)


def reset_only_last_user_message(user_id: int) -> None:
    with _db() as conn:
        row = conn.execute(
            "SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM messages WHERE id = ?", (row[0],))


# ----------------------------------------------------------------------------
# Общие экранные функции (используются и командами, и кнопками клавиатуры)
# ----------------------------------------------------------------------------

WELCOME = (
    "👋 Привет! Я личный бот на моделях GLM (Z.ai).\n\n"
    "Пишите мне — отвечаю с эффектом живой печати и помню контекст беседы.\n"
    "📷 Фото (можно с подписью-вопросом) — опишу его.\n\n"
    "Кнопки внизу экрана:\n"
    f"• {BTN_PROMPTS} — готовые подсказки (ужин, свидание, перевод…)\n"
    f"• {BTN_MODEL} — выбрать модель (есть бесплатные 🆓)\n"
    f"• {BTN_RESET} — начать диалог заново"
)

HELP_TEXT = (
    "ℹ️ Как пользоваться\n\n"
    "Пишите обычные сообщения — ответ печатается в реальном времени, "
    f"в памяти последние {CONTEXT_MESSAGES} сообщений диалога.\n\n"
    f"💡 {BTN_PROMPTS}: готовые подсказки в один тап. Шаблоны "
    "(«Исправить текст», «Перевести», «Объяснить») попросят прислать текст "
    "следующим сообщением.\n\n"
    f"🧠 {BTN_MODEL}: выбор модели кнопкой. Бесплатные "
    "(GLM-4.7-Flash, GLM-4.6V-Flash) не тратят деньги; платные списывают "
    "копейки за токены.\n\n"
    f"🗑 {BTN_RESET}: забыть историю и начать новую тему.\n\n"
    "📷 Фото уходит бесплатной vision-модели GLM-4.6V-Flash."
)

DENIED_TEXT = "🔒 Это личный бот. Доступ только для его хозяев."


async def show_model_menu(message: Message) -> None:
    current = get_user_model(message.from_user.id)
    await message.answer(
        "🧠 Выберите модель (кнопка с ✅ — текущая):",
        reply_markup=model_keyboard(current),
    )


async def do_reset(message: Message) -> None:
    n = reset_history(message.from_user.id)
    PENDING.pop(message.from_user.id, None)
    await message.answer(f"🗑 История очищена ({n} сообщений). Начинаем с чистого листа!")


async def show_prompts(message: Message) -> None:
    await message.answer(
        "💡 Быстрые подсказки — жмите кнопку, я сразу отвечу.\n"
        "Шаблоны ✍️🌍🧠 попросят прислать текст следующим сообщением.",
        reply_markup=prompts_keyboard(),
    )


# ----------------------------------------------------------------------------
# Обработчики команд
# ----------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    await message.answer(WELCOME, reply_markup=main_reply_kb())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    await message.answer(HELP_TEXT)


@router.message(Command("model"))
async def cmd_model(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    await show_model_menu(message)


@router.message(Command("prompts"))
async def cmd_prompts(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    await show_prompts(message)


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    uid = message.from_user.id
    label = MODELS[get_user_model(uid)][0]
    pending = "ждём текст для шаблона" if uid in PENDING else "нет"
    await message.answer(
        "📊 Статус\n"
        f"Модель: {label}\n"
        f"Сообщений в истории: {history_count(uid)}\n"
        f"Помню последних: {CONTEXT_MESSAGES}\n"
        f"Отложенный ввод: {pending}"
    )


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    await do_reset(message)


@router.callback_query(ModelCB.filter())
async def cb_model(cb: CallbackQuery, callback_data: ModelCB) -> None:
    if not cb.from_user or not is_allowed(cb.from_user.id):
        await cb.answer(DENIED_TEXT, show_alert=True)
        return
    if callback_data.model_id not in MODELS:
        await cb.answer("Неизвестная модель", show_alert=True)
        return
    set_user_model(cb.from_user.id, callback_data.model_id)
    label = MODELS[callback_data.model_id][0]
    await cb.message.edit_reply_markup(reply_markup=model_keyboard(callback_data.model_id))
    await cb.answer(f"Выбрано: {label}", show_alert=False)


@router.callback_query(PromptCB.filter())
async def cb_prompt(cb: CallbackQuery, callback_data: PromptCB) -> None:
    if not cb.from_user or not is_allowed(cb.from_user.id):
        await cb.answer(DENIED_TEXT, show_alert=True)
        return
    uid = cb.from_user.id

    if callback_data.pid == "-":  # ✖️ Закрыть
        try:
            await cb.message.delete()
        except Exception:
            pass
        await cb.answer()
        return

    p = PROMPTS.get(callback_data.pid)
    if not p:
        await cb.answer("Неизвестная подсказка", show_alert=True)
        return
    await cb.answer()

    if "tpl" in p:
        # шаблон: просим прислать текст следующим сообщением
        PENDING[uid] = callback_data.pid
        await cb.message.answer(
            f"{p['label']}\n\nПришлите текст одним сообщением 👇",
            reply_markup=ForceReply(selective=True),
        )
        return

    # готовая подсказка: сразу запускаем диалог
    anchor = await cb.message.answer(f"🔸 {p['label']}")
    await run_prompt(anchor, uid, get_user_model(uid), p["prompt"])


# ----------------------------------------------------------------------------
# Обработчики контента: фото и текст
# ----------------------------------------------------------------------------

@router.message(F.photo)
async def on_photo(message: Message) -> None:
    if not message.from_user or not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    uid = message.from_user.id
    photo = message.photo[-1]  # самое большое разрешение
    if photo.file_size and photo.file_size > MAX_PHOTO_MB * 1024 * 1024:
        await message.answer(f"😬 Фото слишком большое (лимит {MAX_PHOTO_MB} МБ).")
        return

    caption = (message.caption or "Опиши, что на изображении.").strip()
    try:
        async with ChatActionSender(
            bot=message.bot, chat_id=message.chat.id, action=ChatAction.TYPING
        ):
            buf = io.BytesIO()
            await message.bot.download(photo, destination=buf)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content = [
                {"type": "text", "text": caption},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]
            # фото всегда смотрит бесплатная vision-модель
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            api_messages += build_api_messages(uid, VISION_MODEL)
            api_messages.append({"role": "user", "content": content})
            answer = await ask_zai(VISION_MODEL, api_messages)
    except Exception as exc:
        await message.answer(friendly_error(exc))
        return

    # в историю кладём текстовую версию, чтобы не раздувать БД base64-картинками
    add_message(uid, "user", f"[фото] {caption}")
    add_message(uid, "assistant", answer)
    for part in split_text(answer):
        await message.answer(part)


@router.message(F.text)
async def on_text(message: Message) -> None:
    if not message.from_user or not is_allowed(message.from_user.id):
        await message.answer(DENIED_TEXT)
        return
    uid = message.from_user.id
    txt = message.text.strip()

    # кнопки постоянной клавиатуры = команды
    if txt == BTN_PROMPTS:
        await show_prompts(message)
        return
    if txt == BTN_MODEL:
        await show_model_menu(message)
        return
    if txt == BTN_RESET:
        await do_reset(message)
        return

    # отложенный ввод: ждали текст для шаблонной подсказки
    if uid in PENDING:
        pid = PENDING.pop(uid)
        p = PROMPTS.get(pid)
        if p and "tpl" in p:
            prompt = p["tpl"].format(text=txt)
            await run_prompt(message, uid, get_user_model(uid), prompt)
            return

    # обычное сообщение — диалог со стримингом
    await run_prompt(message, uid, get_user_model(uid), txt)


# ----------------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------------

async def main() -> None:
    problems = []
    if not TELEGRAM_BOT_TOKEN:
        problems.append("  • TELEGRAM_BOT_TOKEN — токен от @BotFather")
    if not ZAI_API_KEY:
        problems.append("  • ZAI_API_KEY — ключ с z.ai (меню профиля → API Keys)")
    if problems:
        raise SystemExit(
            "❌ Не заполнен .env. Добавьте:\n" + "\n".join(problems)
        )
    if not ALLOWED_USER_IDS:
        log.warning("ALLOWED_USER_IDS пуст — бот будет отвечать ЛЮБОМУ пользователю!")

    init_db()
    bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    log.info("Бот запущен. Модели: %s", ", ".join(MODELS))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
