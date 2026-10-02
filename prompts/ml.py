# prompts/ml.py

ML_SYSTEM_PROMPT = """
You are FRIDAE's machine-learning analyst.

Your job is to plan and explain supervised tabular ML workflows using the
provided dataset context and deterministic ML tools.

Rules:
1. Never invent dataset columns, target values, metrics, or model results.
2. Identify the target column explicitly before training.
3. Prefer a simple, reproducible baseline before proposing more complex models.
4. Distinguish classification from regression and explain the evidence.
5. Consider missing values, categorical variables, class imbalance, leakage,
   and insufficient sample size.
6. Do not claim a model is production-ready from a single train/test split.
7. Report metrics with their exact names and avoid unsupported comparisons.
8. Use the available ML tools rather than writing arbitrary model-training
   code when the tool can perform the required operation.
9. Keep recommendations tied to the actual dataset and user question.
10. If the evidence is insufficient, say what is missing.

When producing a structured ML plan, use this JSON shape:

{
  "target_column": "column name",
  "task_type": "classification|regression|unknown",
  "objective": "short description",
  "checks": ["check 1", "check 2"],
  "workflow": ["step 1", "step 2"],
  "tool": "train_baseline|none",
  "reason": "brief justification"
}
"""


def build_ml_context(
    *,
    dataset_context: str = "",
    question: str = "",
    target_column: str = "",
    task_type: str = "",
    ml_result: str = "",
) -> str:
    """Build a compact ML-specific context block for the LLM."""
    sections: list[str] = []

    if dataset_context:
        sections.append(f"DATASET CONTEXT:\n{dataset_context}")

    if question:
        sections.append(f"USER QUESTION:\n{question}")

    if target_column:
        sections.append(f"TARGET COLUMN:\n{target_column}")

    if task_type:
        sections.append(f"TASK TYPE:\n{task_type}")

    if ml_result:
        sections.append(f"ML RESULT:\n{ml_result}")

    return "\n\n".join(sections)


def build_ml_plan_messages(
    *,
    dataset_context: str,
    question: str,
    target_column: str = "",
) -> list[dict[str, str]]:
    """Return OpenAI-compatible messages for ML planning."""
    context = build_ml_context(
        dataset_context=dataset_context,
        question=question,
        target_column=target_column,
    )

    return [
        {"role": "system", "content": ML_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                "Create the ML plan using only the supplied evidence.\n\n"
                f"{context}"
            ),
        },
    ]


def build_ml_result_messages(
    *,
    question: str,
    dataset_context: str,
    ml_result: str,
) -> list[dict[str, str]]:
    """Return messages for explaining an observed ML result."""
    context = build_ml_context(
        dataset_context=dataset_context,
        question=question,
        ml_result=ml_result,
    )

    return [
        {"role": "system", "content": ML_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                "Explain the observed ML result concisely. "
                "Do not add metrics or conclusions not present in the evidence.\n\n"
                f"{context}"
            ),
        },
    ]
