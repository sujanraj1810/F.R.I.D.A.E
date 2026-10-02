# state.py

from __future__ import annotations

from typing import Any, TypedDict

from dataset_manager import DatasetManager


class FridaeState(TypedDict, total=False):
    chat_id: str
    question: str
    history: list[dict[str, str]]
    run_id: str
    dataset_manager: DatasetManager

    dataset_name: str
    dataset_format: str
    dataset_context: str
    dataset_profile: dict[str, Any]

    target_column: str
    ml_task: str

    plan: dict[str, Any]
    plan_type: str
    generated_code: str
    tool_name: str
    tool_call: dict[str, Any]
    response_type: str
    model_response: dict[str, Any]

    execution_result: dict[str, Any]
    execution_ok: bool
    verification_status: str
    verification_ok: bool
    repair_feedback: str

    recovery_attempts: int
    analysis_ready: bool
    context_ready: bool
    step_count: int
    deadline: float
    answer_budget: int

    answer: str
    error: str
    artifacts: list[dict[str, Any]]
    events: list[dict[str, Any]]


def initial_state(
    *,
    chat_id: str,
    question: str,
    dataset_manager: DatasetManager,
    history: list[dict[str, str]] | None = None,
    run_id: str = "",
    deadline: float = 0.0,
    answer_budget: int = 210,
) -> FridaeState:
    """Create the canonical initial graph state."""
    return {
        "chat_id": str(chat_id),
        "question": str(question),
        "history": list(history or []),
        "run_id": str(run_id),
        "dataset_manager": dataset_manager,
        "dataset_name": "",
        "dataset_format": "",
        "dataset_context": "",
        "dataset_profile": {},
        "target_column": "",
        "ml_task": "",
        "plan": {},
        "plan_type": "",
        "generated_code": "",
        "tool_name": "",
        "tool_call": {},
        "response_type": "",
        "model_response": {},
        "execution_result": {},
        "execution_ok": False,
        "verification_status": "pending",
        "verification_ok": False,
        "repair_feedback": "",
        "recovery_attempts": 0,
        "analysis_ready": False,
        "context_ready": False,
        "step_count": 0,
        "deadline": float(deadline),
        "answer_budget": int(answer_budget),
        "answer": "",
        "artifacts": [],
        "error": "",
        "events": [],
    }


def state_snapshot(state: FridaeState) -> dict[str, Any]:
    """Return a log-safe snapshot without serializing the live manager."""
    return {
        "chat_id": state.get("chat_id", ""),
        "question": state.get("question", ""),
        "run_id": state.get("run_id", ""),
        "dataset_name": state.get("dataset_name", ""),
        "dataset_format": state.get("dataset_format", ""),
        "target_column": state.get("target_column", ""),
        "ml_task": state.get("ml_task", ""),
        "plan_type": state.get("plan_type", ""),
        "execution_ok": state.get("execution_ok", False),
        "verification_status": state.get("verification_status", ""),
        "verification_ok": state.get("verification_ok", False),
        "artifacts": state.get("artifacts", []),
        "recovery_attempts": state.get("recovery_attempts", 0),
        "step_count": state.get("step_count", 0),
        "answer": state.get("answer", ""),
        "error": state.get("error", ""),
    }
