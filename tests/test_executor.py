from executor import ValidationError, run_python, validate_python


def test_guardrail_blocks_filesystem_and_dynamic_execution():
    for code in ["import os", "open('x.txt', 'w')", "eval('1+1')"]:
        try:
            validate_python(code)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Guardrail allowed unsafe code: {code}")


def test_python_execution_and_output_cap():
    result = run_python("print(sum([1, 2, 3]))", timeout=2)
    assert result.ok
    assert result.output.strip() == "6"


def test_artifact_helper_is_available_without_filesystem_access():
    captured = []

    def save_text(value, filename="result.txt"):
        captured.append((str(value), filename))
        return {"artifact_id": "abc", "filename": filename}

    result = run_python(
        "artifact = save_text_artifact('hello', 'result.txt')\nprint(artifact['filename'])",
        artifact_helpers={"save_text_artifact": save_text},
        timeout=2,
    )
    assert result.ok
    assert result.output.strip() == "result.txt"
    assert captured == [("hello", "result.txt")]
