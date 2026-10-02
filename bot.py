# bot.py

from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from artifact_manager import ArtifactManager
from config import (
    ARTIFACT_EXPIRY_SECONDS,
    ARTIFACT_ROOT,
    BOT_TOKEN,
    EVAL_TOKEN,
    LOG_PATH,
    MAX_ARTIFACT_BYTES,
    MAX_UPLOAD_BYTES,
)
from dataset_manager import DatasetManager
from graph import graph
from memory import memory_store
from nodes import run_initial_state
from observability import configure_logging, log_event


configure_logging()

app = FastAPI(title="FRIDAE V2")

dataset_manager = DatasetManager()
artifact_manager = ArtifactManager(
    root=ARTIFACT_ROOT,
    max_bytes=MAX_ARTIFACT_BYTES,
    expiry_seconds=ARTIFACT_EXPIRY_SECONDS,
)
telegram_application: Application | None = None
telegram_thread: threading.Thread | None = None
telegram_loop: asyncio.AbstractEventLoop | None = None
worker_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="fridae-worker")

_started = False
_start_lock = threading.Lock()


class DebugRequest(BaseModel):
    question: str
    chat_id: str = "debug"


def _answer_from_state(state: dict[str, Any]) -> str:
    answer = str(state.get("answer", "") or "").strip()
    if answer:
        return answer

    if state.get("error"):
        return "I couldn't complete the analysis. Please check the request and try again."

    return "I couldn't produce an answer."


def _run_graph_sync(chat_id: str, question: str) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    initial = run_initial_state(
        chat_id=chat_id,
        question=question,
        dataset_manager=dataset_manager,
        history=memory_store.get_history(chat_id),
        run_id=run_id,
    )

    try:
        final_state = graph.invoke(
            initial,
            config={"recursion_limit": 50},
        )
        answer = _answer_from_state(final_state)
        artifacts = list(final_state.get("artifacts") or [])

        memory_store.append(chat_id, {"role": "user", "content": question})
        memory_store.append(chat_id, {"role": "assistant", "content": answer})

        log_event(
            "run_completed",
            chat_id=chat_id,
            run_id=run_id,
            answer_length=len(answer),
            artifacts=len(artifacts),
            error=bool(final_state.get("error")),
        )
        return {"answer": answer, "artifacts": artifacts, "run_id": run_id}

    except Exception as exc:
        log_event(
            "run_failed",
            chat_id=chat_id,
            run_id=run_id,
            error=type(exc).__name__,
        )
        return {
            "answer": "I couldn't complete the analysis. An internal error occurred.",
            "artifacts": [],
            "run_id": run_id,
        }


async def _handle_text(chat_id: str, question: str) -> dict[str, Any]:
    lock = memory_store.get_lock(chat_id)

    with lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            worker_pool,
            _run_graph_sync,
            chat_id,
            question,
        )


async def _deliver_artifacts(message: Any, chat_id: str, metadata: list[dict[str, Any]]) -> None:
    for item in metadata:
        artifact_id = str(item.get("artifact_id", ""))
        if not artifact_id:
            continue

        artifact = artifact_manager.get(chat_id, artifact_id)
        if artifact is None or not artifact.path.exists():
            await message.reply_text(
                f"I created `{item.get('filename', 'an artifact')}`, but it expired before delivery.",
                parse_mode="Markdown",
            )
            continue

        try:
            with artifact.path.open("rb") as handle:
                await message.reply_document(
                    document=handle,
                    filename=artifact.filename,
                    caption=f"📎 {artifact.filename}",
                )
        except Exception as exc:
            log_event(
                "artifact_delivery_error",
                chat_id=chat_id,
                artifact_id=artifact_id,
                error=type(exc).__name__,
            )
            await message.reply_text(
                f"I created `{artifact.filename}`, but couldn't send it.",
                parse_mode="Markdown",
            )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    await update.message.reply_text(
        "FRIDAE V2 online. Send me a data-analysis question or upload a CSV/JSON/JSONL dataset."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return

    chat_id = str(update.effective_chat.id)
    memory_store.clear(chat_id)
    dataset_manager.remove(chat_id)
    for artifact in artifact_manager.list(chat_id):
        artifact_manager.remove(chat_id, artifact.artifact_id)

    await update.message.reply_text("Chat history, dataset, and generated artifacts cleared.")


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return

    document = update.message.document
    if document is None:
        return

    chat_id = str(update.effective_chat.id)

    if document.file_size and document.file_size > MAX_UPLOAD_BYTES:
        await update.message.reply_text(
            f"File is too large. Maximum supported upload is {MAX_UPLOAD_BYTES // 1_000_000} MB."
        )
        return

    name = document.file_name or "uploaded_dataset"

    try:
        telegram_file = await document.get_file()
        data = await telegram_file.download_as_bytearray()

        if len(data) > MAX_UPLOAD_BYTES:
            await update.message.reply_text("Uploaded file exceeds the size limit.")
            return

        info = dataset_manager.register_bytes(
            chat_id=chat_id,
            original_name=name,
            data=bytes(data),
        )

        await update.message.reply_text(
            f"Loaded `{info.original_name}` ({info.format.upper()}). You can now ask questions about it.",
            parse_mode="Markdown",
        )
        log_event(
            "dataset_uploaded",
            chat_id=chat_id,
            filename=name,
            size_bytes=len(data),
            format=info.format,
        )

    except Exception as exc:
        log_event(
            "dataset_upload_error",
            chat_id=chat_id,
            filename=name,
            error=type(exc).__name__,
        )
        await update.message.reply_text(
            "I couldn't load that dataset. Please check that it is a supported CSV/JSON/JSONL file."
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = str(update.effective_chat.id)

    try:
        result = await _handle_text(chat_id, text)
        await update.message.reply_text(result["answer"])
        await _deliver_artifacts(update.message, chat_id, result.get("artifacts", []))
    except Exception as exc:
        log_event(
            "telegram_handler_error",
            chat_id=chat_id,
            error=type(exc).__name__,
        )
        await update.message.reply_text("Something went wrong while processing that request.")


async def _telegram_main() -> None:
    global telegram_application

    if not BOT_TOKEN:
        log_event("telegram_disabled", reason="BOT_TOKEN not configured")
        return

    telegram_application = Application.builder().token(BOT_TOKEN).build()
    telegram_application.add_handler(CommandHandler("start", start_command))
    telegram_application.add_handler(CommandHandler("clear", clear_command))
    telegram_application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    telegram_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    await telegram_application.initialize()
    await telegram_application.start()
    await telegram_application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    log_event("telegram_started")

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await telegram_application.updater.stop()
        await telegram_application.stop()
        await telegram_application.shutdown()


def _telegram_thread_target() -> None:
    global telegram_loop
    telegram_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(telegram_loop)
    try:
        telegram_loop.run_until_complete(_telegram_main())
    except Exception as exc:
        log_event("telegram_thread_error", error=type(exc).__name__)
    finally:
        telegram_loop.close()


def start_telegram_once() -> None:
    global _started, telegram_thread
    with _start_lock:
        if _started:
            return
        if not BOT_TOKEN:
            _started = True
            log_event("telegram_not_started", reason="BOT_TOKEN missing")
            return
        telegram_thread = threading.Thread(
            target=_telegram_thread_target,
            name="fridae-telegram",
            daemon=True,
        )
        telegram_thread.start()
        _started = True


@app.on_event("startup")
async def startup_event() -> None:
    start_telegram_once()
    removed = artifact_manager.cleanup_expired()
    log_event("artifact_cleanup", removed=removed)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    worker_pool.shutdown(wait=False, cancel_futures=True)


@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "FRIDAE", "version": "v2", "status": "online"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "telegram_started": _started and telegram_application is not None}


@app.get("/run.jsonl")
async def run_log(x_eval_token: str | None = Header(default=None)) -> FileResponse:
    if not EVAL_TOKEN or x_eval_token != EVAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    path = Path(LOG_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    return FileResponse(path=str(path), media_type="application/x-ndjson", filename=path.name)


@app.post("/debug/ask")
async def debug_ask(request: DebugRequest, x_eval_token: str | None = Header(default=None)) -> dict[str, Any]:
    if not EVAL_TOKEN or x_eval_token != EVAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await _handle_text(str(request.chat_id), request.question.strip())
    return {"answer": result["answer"], "artifacts": result.get("artifacts", [])}


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots() -> str:
    return "User-agent: *\nDisallow: /debug/\nDisallow: /run.jsonl\n"
