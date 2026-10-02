# nodes.py
#
# LangGraph node implementations for FRIDAE V2.
# This version keeps dataset ownership in the graph state rather than creating
# a second DatasetManager registry inside this module.

from __future__ import annotations

import json
import time

import numpy as np
from typing import Any

from config import (
    ANSWER_BUDGET,
    MAX_AGENT_STEPS,
    MAX_RECOVERY_ATTEMPTS,
    ARTIFACT_ROOT,
    ARTIFACT_EXPIRY_SECONDS,
    MAX_ARTIFACT_BYTES,
)
from artifact_manager import ArtifactManager
from dataset_manager import DatasetManager
from executor import ExecutionResult, run_python
from llm import LLMError, chat_completion, extract_json
from memory import memory_store
from prompts.analyst import SYSTEM_PROMPT, build_user_context
from prompts.ml import build_ml_plan_messages, build_ml_result_messages
from tools.data_tools import profile_dataframe
from tools.ml_tools import infer_task, train_baseline
from tools.web_tools import (
    fetch_csv,
    fetch_excel,
    fetch_json,
    fetch_soup,
    fetch_table,
    fetch_url,
)
from observability import log_event
from state import FridaeState


def _deadline_ok(state: FridaeState) -> bool:
    deadline = float(state.get("deadline", 0.0) or 0.0)
    return deadline <= 0 or time.monotonic() < deadline


def _bump_step(state: FridaeState) -> FridaeState:
    state["step_count"] = int(state.get("step_count", 0)) + 1
    return state


def _set_error(state: FridaeState, message: str) -> FridaeState:
    state["error"] = str(message)[:2000]
    return state


_artifact_manager = ArtifactManager(
    root=ARTIFACT_ROOT,
    max_bytes=MAX_ARTIFACT_BYTES,
    expiry_seconds=ARTIFACT_EXPIRY_SECONDS,
)


def _artifact_helpers(state: FridaeState, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    chat_id = str(state.get("chat_id", ""))

    def _record(artifact: Any) -> dict[str, Any]:
        metadata = artifact.as_dict()
        artifacts.append(metadata)
        return metadata

    def save_dataframe(dataframe: Any, filename: str = "analysis.csv", format: str = "csv") -> dict[str, Any]:
        return _record(
            _artifact_manager.write_dataframe(
                chat_id, filename, dataframe, format=format
            )
        )

    def save_text(text: Any, filename: str = "analysis.txt") -> dict[str, Any]:
        return _record(
            _artifact_manager.write_text(chat_id, filename, str(text))
        )

    def save_json(value: Any, filename: str = "analysis.json") -> dict[str, Any]:
        return _record(
            _artifact_manager.write_json(chat_id, filename, value)
        )

    def save_csv(rows: Any, filename: str = "analysis.csv") -> dict[str, Any]:
        return _record(
            _artifact_manager.write_csv(chat_id, filename, rows)
        )

    return {
        "save_dataframe_artifact": save_dataframe,
        "save_text_artifact": save_text,
        "save_json_artifact": save_json,
        "save_csv_artifact": save_csv,
    }


def _dataset_manager(state: FridaeState) -> DatasetManager:
    manager = state.get("dataset_manager")
    if isinstance(manager, DatasetManager):
        return manager
    raise RuntimeError("DatasetManager is missing from graph state.")


def _llm_messages(
    state: FridaeState,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    history = list(state.get("history") or [])
    context = build_user_context(
        question=str(state.get("question", "")),
        dataset_context=str(state.get("dataset_context", "")),
        history=None,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()}
    ]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": context})
    return messages


def _model_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
) -> dict[str, Any]:
    raw = chat_completion(messages, temperature=temperature)
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise LLMError("Model returned a non-object JSON response.")
    return parsed


def _is_ml_request(state: FridaeState) -> bool:
    text = str(state.get("question", "")).lower()
    keywords = (
        "machine learning",
        "ml model",
        "train a model",
        "train model",
        "predict",
        "prediction",
        "classification",
        "classifier",
        "regression",
        "accuracy",
        "f1 score",
        "rmse",
        "mae",
        "r2",
        "feature importance",
        "model performance",
    )
    return any(keyword in text for keyword in keywords)


def prepare_context(state: FridaeState) -> FridaeState:
    _bump_step(state)

    if not _deadline_ok(state):
        return _set_error(state, "Analysis deadline exceeded before context preparation.")

    manager = _dataset_manager(state)
    chat_id = str(state.get("chat_id", ""))

    try:
        dataset = manager.get(chat_id)
        if dataset is not None:
            profile = manager.context(chat_id)
            state["dataset_profile"] = profile
            state["dataset_context"] = json.dumps(
                profile,
                ensure_ascii=False,
                default=str,
            )
            state["dataset_name"] = dataset.original_name
            state["dataset_format"] = dataset.format
        else:
            state["dataset_profile"] = {}
            state["dataset_context"] = ""
            state["dataset_name"] = ""
            state["dataset_format"] = ""

        state["context_ready"] = True
        log_event(
            "context_prepared",
            chat_id=chat_id,
            run_id=state.get("run_id"),
            has_dataset=dataset is not None,
        )
    except Exception as exc:
        log_event(
            "context_error",
            chat_id=chat_id,
            run_id=state.get("run_id"),
            error=str(exc),
        )
        _set_error(state, f"Context preparation failed: {exc}")

    return state


def create_plan(state: FridaeState) -> FridaeState:
    _bump_step(state)

    if state.get("error"):
        return state

    if not _deadline_ok(state):
        return _set_error(state, "Analysis deadline exceeded while creating a plan.")

    try:
        if _is_ml_request(state) and state.get("dataset_context"):
            target = str(state.get("target_column", "") or "")
            messages = build_ml_plan_messages(
                dataset_context=str(state.get("dataset_context", "")),
                question=str(state.get("question", "")),
                target_column=target,
            )
            plan = _model_json(messages)
            state["plan"] = plan
            state["plan_type"] = "ml"
        else:
            plan = _model_json(_llm_messages(state))
            state["plan"] = plan
            state["plan_type"] = "analysis"

        log_event(
            "plan_created",
            chat_id=str(state.get("chat_id", "")),
            run_id=state.get("run_id"),
            plan_type=state.get("plan_type"),
        )
    except Exception as exc:
        log_event(
            "plan_error",
            chat_id=str(state.get("chat_id", "")),
            run_id=state.get("run_id"),
            error=str(exc),
        )
        _set_error(state, f"Planning failed: {exc}")

    return state


def _resolve_ml_target(state: FridaeState) -> str:
    explicit = str(state.get("target_column", "") or "").strip()
    if explicit:
        return explicit

    plan = state.get("plan") or {}
    target = str(plan.get("target_column", "") or "").strip()
    if target:
        state["target_column"] = target
        return target

    raise ValueError(
        "No target column was identified. Please specify the column to predict."
    )


def generate_analysis(state: FridaeState) -> FridaeState:
    _bump_step(state)

    if state.get("error"):
        return state

    if not _deadline_ok(state):
        return _set_error(state, "Analysis deadline exceeded during generation.")

    # ML plans use deterministic ML tools rather than asking the LLM to emit
    # arbitrary training code.
    if state.get("plan_type") == "ml":
        try:
            target = _resolve_ml_target(state)
            manager = _dataset_manager(state)
            df = manager.load(str(state.get("chat_id", "")))

            task = infer_task(df, target)
            state["ml_task"] = task.task_type
            state["target_column"] = target
            state["generated_code"] = ""
            state["tool_name"] = "train_baseline"
            state["analysis_ready"] = True
            return state
        except Exception as exc:
            log_event(
                "ml_generation_error",
                chat_id=str(state.get("chat_id", "")),
                run_id=state.get("run_id"),
                error=str(exc),
            )
            return _set_error(state, f"ML preparation failed: {exc}")

    try:
        messages = _llm_messages(state)
        result = _model_json(messages)
        state["model_response"] = result
        state["response_type"] = str(result.get("type", "") or "").strip()

        code = str(result.get("code", "") or "").strip()
        tool_call = result.get("tool_call")

        if code:
            state["generated_code"] = code
            state["tool_name"] = "run_python"
            state["analysis_ready"] = True
        elif isinstance(tool_call, dict):
            # Web/data retrieval remains helper-level. The graph does not
            # expose an unbounded external tool router in this phase.
            state["tool_call"] = tool_call
            state["generated_code"] = ""
            state["analysis_ready"] = False
        else:
            state["generated_code"] = ""
            state["analysis_ready"] = False
            state["answer"] = str(result.get("answer", "") or "").strip()

    except Exception as exc:
        log_event(
            "generation_error",
            chat_id=str(state.get("chat_id", "")),
            run_id=state.get("run_id"),
            error=str(exc),
        )
        _set_error(state, f"Analysis generation failed: {exc}")

    return state


def _retrieval_helpers() -> dict[str, Any]:
    return {
        "fetch_url": fetch_url,
        "fetch_soup": fetch_soup,
        "fetch_table": fetch_table,
        "fetch_csv": fetch_csv,
        "fetch_excel": fetch_excel,
        "fetch_json": fetch_json,
    }


def execute_analysis(state: FridaeState) -> FridaeState:
    _bump_step(state)

    if state.get("error"):
        return state

    if not _deadline_ok(state):
        return _set_error(state, "Analysis deadline exceeded before execution.")

    if state.get("plan_type") == "ml":
        try:
            target = _resolve_ml_target(state)
            manager = _dataset_manager(state)
            df = manager.load(str(state.get("chat_id", "")))
            result = train_baseline(df, target)

            state["execution_result"] = {
                "status": "OK",
                "task_type": result.task_type,
                "target_column": result.target_column,
                "metrics": result.metrics,
                "prediction_count": len(result.predictions),
                "feature_importance": result.feature_importance,
            }
            state["verification_status"] = "pending"
            state["execution_ok"] = True
            return state
        except Exception as exc:
            state["execution_ok"] = False
            state["execution_result"] = {
                "status": "ERROR",
                "error": str(exc)[:2000],
            }
            log_event(
                "ml_execution_error",
                chat_id=str(state.get("chat_id", "")),
                run_id=state.get("run_id"),
                error=str(exc),
            )
            return state

    code = str(state.get("generated_code", "") or "").strip()
    if not code:
        state["execution_ok"] = True
        state["execution_result"] = {
            "status": "NO_EXECUTION",
            "answer": state.get("answer", ""),
        }
        return state

    manager = _dataset_manager(state)
    chat_id = str(state.get("chat_id", ""))
    artifacts: list[dict[str, Any]] = []

    try:
        dataset_loader = lambda: manager.load(chat_id)
        remaining = max(1.0, float(state.get("deadline", time.monotonic() + 60)) - time.monotonic())
        result: ExecutionResult = run_python(
            code,
            dataset_loader=dataset_loader,
            retrieval_helpers=_retrieval_helpers(),
            artifact_helpers={
                **_artifact_helpers(state, artifacts),
                "profile_dataframe": profile_dataframe,
            },
            timeout=min(60.0, remaining),
        )

        state["execution_result"] = {
            "status": "OK" if result.ok else "TIMEOUT" if result.timed_out else "ERROR",
            "stdout": result.output,
            "stderr": "",
            "value": result.output.strip() if result.output.strip() else None,
            "error": result.error,
            "elapsed_seconds": None,
        }
        state["artifacts"] = artifacts
        state["execution_ok"] = result.ok
        state["verification_status"] = "pending"

        log_event(
            "analysis_executed",
            chat_id=chat_id,
            run_id=state.get("run_id"),
            status=state["execution_result"]["status"],
            artifacts=len(artifacts),
        )
    except Exception as exc:
        state["execution_ok"] = False
        state["execution_result"] = {
            "status": "ERROR",
            "error": str(exc)[:2000],
        }
        log_event(
            "execution_error",
            chat_id=chat_id,
            run_id=state.get("run_id"),
            error=type(exc).__name__,
        )

    return state

def verify_result(state: FridaeState) -> FridaeState:
    _bump_step(state)

    if state.get("error"):
        state["verification_status"] = "failed"
        return state

    result = state.get("execution_result") or {}
    status = str(result.get("status", ""))

    if state.get("plan_type") == "ml":
        metrics = result.get("metrics")
        valid = (
            status == "OK"
            and isinstance(metrics, dict)
            and bool(metrics)
            and all(
                isinstance(value, (int, float))
                and np.isfinite(float(value))
                for value in metrics.values()
            )
        )
    else:
        valid = status == "OK"

    state["verification_status"] = "passed" if valid else "failed"
    state["verification_ok"] = bool(valid)

    log_event(
        "result_verified",
        chat_id=str(state.get("chat_id", "")),
        run_id=state.get("run_id"),
        status=state["verification_status"],
    )
    return state


def repair_analysis(state: FridaeState) -> FridaeState:
    _bump_step(state)

    attempts = int(state.get("recovery_attempts", 0)) + 1
    state["recovery_attempts"] = attempts

    if attempts > MAX_RECOVERY_ATTEMPTS:
        return _set_error(
            state,
            "The analysis could not be completed after the allowed recovery attempts.",
        )

    if state.get("plan_type") == "ml":
        error = str((state.get("execution_result") or {}).get("error", ""))
        if "target column" in error.lower():
            return _set_error(state, error)

        # Re-run ML preparation on the next graph cycle. The deterministic
        # baseline itself remains unchanged; repair is bounded.
        state["execution_ok"] = False
        state["verification_status"] = "failed"
        state["analysis_ready"] = True
        return state

    result = state.get("execution_result") or {}
    error = result.get("error") or result.get("stderr") or "Unknown execution failure."

    state["repair_feedback"] = str(error)[:3000]

    try:
        repair_prompt = (
            "Repair the previous analysis code. Return JSON only with a `code` "
            "field containing corrected Python. Keep the original user goal. "
            "Do not guess values and do not use filesystem/network primitives "
            "outside the supplied helper functions.\n\n"
            f"QUESTION:\n{state.get('question', '')}\n\n"
            f"PREVIOUS CODE:\n{state.get('generated_code', '')}\n\n"
            f"ERROR:\n{state['repair_feedback']}"
        )
        result_json = _model_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": repair_prompt},
            ]
        )
        code = str(result_json.get("code", "") or "").strip()
        if not code:
            return _set_error(state, "Repair produced no executable code.")

        state["generated_code"] = code
        state["execution_ok"] = False
        state["verification_status"] = "pending"
        state["analysis_ready"] = True
    except Exception as exc:
        return _set_error(state, f"Repair failed: {exc}")

    return state


def finalize(state: FridaeState) -> FridaeState:
    _bump_step(state)

    if state.get("answer"):
        answer = str(state["answer"]).strip()
        state["answer"] = answer[: int(state.get("answer_budget", 210) or 210)]
        return state

    if state.get("error"):
        state["answer"] = f"I couldn't complete the analysis: {state['error']}"
        return state

    if state.get("plan_type") == "ml":
        result = state.get("execution_result") or {}
        explanation = ""

        try:
            messages = build_ml_result_messages(
                question=str(state.get("question", "")),
                dataset_context=str(state.get("dataset_context", "")),
                ml_result=json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                ),
            )
            raw = chat_completion(messages, temperature=0.0)
            explanation = raw.strip()
        except Exception:
            explanation = ""

        if not explanation:
            explanation = (
                f"Trained a {result.get('task_type', 'baseline')} baseline on "
                f"`{result.get('target_column', 'the target')}`. "
                f"Metrics: {json.dumps(result.get('metrics', {}), default=str)}."
            )

        state["answer"] = explanation[: int(state.get("answer_budget", 210) or 210)]
        return state

    result = state.get("execution_result") or {}
    value = result.get("value")
    stdout = str(result.get("stdout", "") or "").strip()

    if value is not None:
        answer = str(value)
    elif stdout:
        answer = stdout
    else:
        answer = "Analysis completed successfully."

    state["answer"] = answer[: int(state.get("answer_budget", 210) or 210)]
    return state


def run_initial_state(
    *,
    chat_id: str,
    question: str,
    dataset_manager: DatasetManager,
    history: list[dict[str, str]] | None = None,
    run_id: str = "",
) -> FridaeState:
    now = time.monotonic()
    return {
        "chat_id": str(chat_id),
        "question": str(question),
        "history": list(history or memory_store.get_history(chat_id)),
        "dataset_manager": dataset_manager,
        "dataset_name": "",
        "dataset_format": "",
        "dataset_context": "",
        "dataset_profile": {},
        "target_column": "",
        "plan": {},
        "plan_type": "",
        "generated_code": "",
        "tool_name": "",
        "tool_call": {},
        "model_response": {},
        "execution_result": {},
        "execution_ok": False,
        "verification_status": "pending",
        "verification_ok": False,
        "artifacts": [],
        "repair_feedback": "",
        "recovery_attempts": 0,
        "analysis_ready": False,
        "context_ready": False,
        "ml_task": "",
        "step_count": 0,
        "deadline": now + ANSWER_BUDGET,
        "answer_budget": ANSWER_BUDGET,
        "answer": "",
        "error": "",
        "run_id": run_id,
        "events": [],
    }
