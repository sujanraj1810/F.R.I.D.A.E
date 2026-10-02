# graph.py

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from config import MAX_AGENT_STEPS, MAX_RECOVERY_ATTEMPTS
from nodes import (
    create_plan,
    execute_analysis,
    finalize,
    generate_analysis,
    prepare_context,
    repair_analysis,
    verify_result,
)
from state import FridaeState


def _step_limit_reached(state: FridaeState) -> bool:
    return int(state.get("step_count", 0)) >= MAX_AGENT_STEPS


def _route_after_plan(
    state: FridaeState,
) -> Literal["generate_analysis", "finalize"]:
    if _step_limit_reached(state):
        return "finalize"

    if state.get("error"):
        return "finalize"

    if state.get("response_type") == "final" and state.get("answer"):
        return "finalize"

    return "generate_analysis"


def _route_after_generation(
    state: FridaeState,
) -> Literal["execute_analysis", "finalize"]:
    if _step_limit_reached(state):
        return "finalize"

    if state.get("error"):
        return "finalize"

    if state.get("response_type") == "final" and state.get("answer"):
        return "finalize"

    # ML analysis uses a deterministic tool rather than generated Python.
    # Therefore generated_code is intentionally empty on the ML path.
    if state.get("plan_type") == "ml" and state.get("ml_task"):
        return "execute_analysis"

    # Tool-call routing is deliberately not implemented as a separate graph
    # branch yet. Retrieval helpers are exposed to generated Python so the
    # analyst can perform retrieval within the controlled execution step.
    if not state.get("generated_code"):
        state["error"] = "The analysis stage did not produce executable code."
        return "finalize"

    return "execute_analysis"


def _route_after_execution(
    state: FridaeState,
) -> Literal["verify_result", "repair_analysis", "finalize"]:
    if _step_limit_reached(state):
        return "finalize"

    if state.get("execution_ok"):
        return "verify_result"

    if int(state.get("recovery_attempts", 0)) < MAX_RECOVERY_ATTEMPTS:
        return "repair_analysis"

    return "finalize"


def _route_after_verification(
    state: FridaeState,
) -> Literal["finalize", "repair_analysis"]:
    if _step_limit_reached(state):
        return "finalize"

    if state.get("verification_ok"):
        return "finalize"

    if int(state.get("recovery_attempts", 0)) < MAX_RECOVERY_ATTEMPTS:
        return "repair_analysis"

    return "finalize"


def _route_after_repair(
    state: FridaeState,
) -> Literal["execute_analysis", "finalize"]:
    if _step_limit_reached(state):
        return "finalize"

    if state.get("error"):
        return "finalize"

    if state.get("plan_type") == "ml" and state.get("ml_task"):
        return "execute_analysis"

    if not state.get("generated_code"):
        state["error"] = "Repair did not produce executable code."
        return "finalize"

    return "execute_analysis"


def build_graph():
    """Build and compile the FRIDAE V2 analysis graph."""
    workflow = StateGraph(FridaeState)

    workflow.add_node("prepare_context", prepare_context)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("generate_analysis", generate_analysis)
    workflow.add_node("execute_analysis", execute_analysis)
    workflow.add_node("verify_result", verify_result)
    workflow.add_node("repair_analysis", repair_analysis)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "prepare_context")
    workflow.add_edge("prepare_context", "create_plan")

    workflow.add_conditional_edges(
        "create_plan",
        _route_after_plan,
        {
            "generate_analysis": "generate_analysis",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "generate_analysis",
        _route_after_generation,
        {
            "execute_analysis": "execute_analysis",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "execute_analysis",
        _route_after_execution,
        {
            "verify_result": "verify_result",
            "repair_analysis": "repair_analysis",
            "finalize": "finalize",
        },
    )

    workflow.add_conditional_edges(
        "verify_result",
        _route_after_verification,
        {
            "finalize": "finalize",
            "repair_analysis": "repair_analysis",
        },
    )

    workflow.add_conditional_edges(
        "repair_analysis",
        _route_after_repair,
        {
            "execute_analysis": "execute_analysis",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


graph = build_graph()
