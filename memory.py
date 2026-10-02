# memory.py

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from config import MAX_AGENT_STEPS


@dataclass
class ChatMemory:
    history: list[dict[str, str]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def get_history(self) -> list[dict[str, str]]:
        with self.lock:
            return [dict(message) for message in self.history]

    def append(self, role: str, content: str) -> None:
        with self.lock:
            self.history.append(
                {"role": str(role), "content": str(content)}
            )
            max_messages = max(10, MAX_AGENT_STEPS * 2)
            if len(self.history) > max_messages:
                self.history = self.history[-max_messages:]

    def clear(self) -> None:
        with self.lock:
            self.history.clear()


class MemoryStore:
    def __init__(self) -> None:
        self._chats: dict[str, ChatMemory] = {}
        self._lock = threading.RLock()

    def _get_or_create(self, chat_id: str) -> ChatMemory:
        key = str(chat_id)
        with self._lock:
            if key not in self._chats:
                self._chats[key] = ChatMemory()
            return self._chats[key]

    def get_history(self, chat_id: str) -> list[dict[str, str]]:
        return self._get_or_create(chat_id).get_history()

    def append(self, chat_id: str, message: dict[str, str]) -> None:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        self._get_or_create(chat_id).append(role, content)

    def clear(self, chat_id: str) -> None:
        with self._lock:
            self._chats.pop(str(chat_id), None)

    def get_lock(self, chat_id: str) -> threading.RLock:
        return self._get_or_create(chat_id).lock

    def snapshot(self, chat_id: str) -> dict[str, Any]:
        memory = self._get_or_create(chat_id)
        return {
            "chat_id": str(chat_id),
            "message_count": len(memory.get_history()),
        }


memory_store = MemoryStore()
