from pathlib import Path

from memory_router import benchmark


def test_mcp_offline_benchmark_cases_file_loads():
    cases_path = Path("benchmarks/mcp_offline_cases.json")
    cases = benchmark.load_cases(cases_path)

    names = {case.name for case in cases}
    assert {"mcp coding prompt", "medium coding prompt"} <= names

    prompt_case = next(case for case in cases if case.name == "mcp coding prompt")
    medium_case = next(case for case in cases if case.name == "medium coding prompt")

    assert "pytest" in prompt_case.query.lower()
    assert prompt_case.session_id == "mcp-test"
    assert prompt_case.memories
    assert "offline-testable" in medium_case.query.lower()
    assert medium_case.session_id == "medium-prompt"
    assert medium_case.memories
