import asyncio
import importlib
import json


def test_mcp_build_context_preserves_large_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    import memory_router.config as config_mod
    import memory_router.context_builder as context_builder_mod
    import memory_router.memory.sqlite_store as sqlite_store_mod
    import memory_router.mcp_server as mcp_server_mod

    config_mod = importlib.reload(config_mod)
    sqlite_store_mod = importlib.reload(sqlite_store_mod)
    context_builder_mod = importlib.reload(context_builder_mod)
    mcp_server_mod = importlib.reload(mcp_server_mod)

    prompt = "\n".join(
        [
            "Design a very large MCP code review for offline coding workflows.",
            "Cover classifier routing, token budgets, memory ranking, working memory,",
            "mycelium retrieval, decay, consolidation, importer behavior, safety,",
            "benchmarking, CLI compatibility, and test strategy.",
            "",
            "Repeatable checklist:",
        ]
        + [
            f"- Section {i}: explain exact file-level behavior, failure modes, and tests."
            for i in range(1, 801)
        ]
    )

    assert len(prompt) > 30_000

    async def run():
        server = mcp_server_mod._create_server()
        result = await server.call_tool(
            "build_context",
            {
                "query": prompt,
                "session_id": "large-prompt",
                "use_memory": False,
            },
        )
        return json.loads(result[0].text)

    data = asyncio.run(run())

    assert data["messages"][-1]["content"] == prompt
    assert data["classification"]["domain"] == "software"
