"""Tests for Obsidian Markdown rendering, parsing, and round-trip fidelity."""

from __future__ import annotations

from memory_router.memory.sqlite_store import Memory
from memory_router.memory.obsidian.models import ConceptLink, KnowledgeNote, category_for
from memory_router.memory.obsidian.renderer import (
    markdown_to_memory,
    memory_to_markdown,
    parse_frontmatter,
    render_knowledge_note,
    serialize_frontmatter,
    wikilink,
)
from memory_router.memory.obsidian.utils import (
    from_iso,
    redact,
    safe_join,
    sanitize_filename,
    slugify,
    to_iso,
)


class TestFrontmatter:
    def test_roundtrip(self):
        data = {"id": 1, "task": "explain", "concepts": ["a", "b"]}
        text = serialize_frontmatter(data) + "\n\nbody here"
        parsed, body = parse_frontmatter(text)
        assert parsed == data
        assert body == "body here"

    def test_no_frontmatter(self):
        parsed, body = parse_frontmatter("just text, no fence")
        assert parsed == {}
        assert body == "just text, no fence"

    def test_malformed_yaml_is_safe(self):
        # Non-dict frontmatter falls back to {}
        parsed, _ = parse_frontmatter("---\n- a\n- b\n---\nbody")
        assert parsed == {}


class TestWikilink:
    def test_plain(self):
        assert wikilink("CatBoost") == "[[CatBoost]]"

    def test_with_weight(self):
        assert wikilink("CatBoost", 0.82) == "[[CatBoost]] <!-- weight 0.82 -->"

    def test_zero_weight_omitted(self):
        assert wikilink("CatBoost", 0.0) == "[[CatBoost]]"


class TestMemoryRoundTrip:
    def _mem(self):
        return Memory(
            id=123,
            task="explain",
            domain="software",
            concepts=["RideCompare", "CatBoost", "Flutter"],
            content="RideCompare uses CatBoost for fare prediction.\nMulti-line body.",
            importance=0.9,
            confidence=0.86,
            memory_type="semantic",
            source="user",
            created_at=1_750_000_000.0,
            last_used=1_750_000_500.0,
            usage_count=4,
        )

    def test_lossless_content(self):
        mem = self._mem()
        md, _ = memory_to_markdown(mem, neighbors=[(88, 0.74)])
        back = markdown_to_memory(md)
        assert back.content == mem.content

    def test_all_scalar_fields_preserved(self):
        mem = self._mem()
        md, _ = memory_to_markdown(mem)
        back = markdown_to_memory(md)
        assert back.id == 123
        assert back.task == "explain"
        assert back.domain == "software"
        assert back.concepts == ["RideCompare", "CatBoost", "Flutter"]
        assert back.importance == 0.9
        assert back.confidence == 0.86
        assert back.memory_type == "semantic"
        assert back.source == "user"
        assert back.usage_count == 4

    def test_timestamps_iso_then_back(self):
        mem = self._mem()
        md, _ = memory_to_markdown(mem)
        assert "2025-" in md or "2026-" in md  # ISO rendered
        back = markdown_to_memory(md)
        # within rounding of seconds
        assert abs(back.created_at - mem.created_at) < 2

    def test_backlinks_rendered(self):
        mem = self._mem()
        md, _ = memory_to_markdown(mem, neighbors=[(88, 0.74), (91, 0.61)])
        assert "## Linked Memories" in md
        assert "[[mem_88]] <!-- weight 0.74 -->" in md
        assert "[[mem_91]]" in md

    def test_backlinks_disabled(self):
        mem = self._mem()
        md, _ = memory_to_markdown(mem, neighbors=[(88, 0.74)], generate_backlinks=False)
        assert "Linked Memories" not in md

    def test_redaction_removes_secret(self):
        mem = self._mem()
        mem.content = "deploy key sk-abcdefghijklmnopqrstuvwxyz12345"
        md, n = memory_to_markdown(mem, redact=True)
        assert n >= 1
        assert "sk-abcdefghijklmnopqrstuvwxyz12345" not in md
        assert "[REDACTED]" in md


class TestKnowledgeNote:
    def test_render_has_sections(self):
        note = KnowledgeNote(
            title="RideCompare",
            category="01_Projects",
            domain="software",
            summary="A fare prediction app.",
            key_learnings=["Uses CatBoost", "Flutter frontend"],
            related_concepts=[ConceptLink("CatBoost", 0.82), ConceptLink("Flutter", 0.74)],
            source_memory_ids=[123, 456],
        )
        md = render_knowledge_note(note)
        assert "# RideCompare" in md
        assert "## Summary" in md
        assert "## Key Learnings" in md
        assert "## Related Concepts" in md
        assert "[[CatBoost]] <!-- weight 0.82 -->" in md
        assert "## Source Memories" in md
        assert "[[mem_123]]" in md

    def test_empty_summary_placeholder(self):
        note = KnowledgeNote(title="X", category="00_Inbox", domain="general")
        md = render_knowledge_note(note)
        assert "_No summary available._" in md

    def test_rel_path_sanitized(self):
        note = KnowledgeNote(title="A/B: weird*name", category="02_Research", domain="d")
        assert ".." not in note.rel_path
        assert note.rel_path.startswith("02_Research/")
        assert note.rel_path.endswith(".md")


class TestCategoryFor:
    def test_software_domain(self):
        assert category_for("software") == "01_Projects"

    def test_research_domains(self):
        assert category_for("ml") == "02_Research"
        assert category_for("finance") == "02_Research"
        assert category_for("legal") == "02_Research"
        assert category_for("medical") == "02_Research"

    def test_general_falls_back_to_task(self):
        # The classifier's default domain must not all dump into Inbox.
        assert category_for("general", "explain") == "02_Research"
        assert category_for("general", "code") == "01_Projects"
        assert category_for("general", "agentic") == "03_Decisions"
        assert category_for("general", "chat") == "05_Conversations"

    def test_general_no_task_is_inbox(self):
        assert category_for("general", "general") == "00_Inbox"
        assert category_for("general", "") == "00_Inbox"

    def test_empty_domain_is_inbox(self):
        assert category_for("", "") == "00_Inbox"

    def test_domain_wins_over_task(self):
        # A real domain should not be overridden by task fallback.
        assert category_for("software", "explain") == "01_Projects"


class TestUtils:
    def test_sanitize_strips_traversal(self):
        assert "/" not in sanitize_filename("../../etc/passwd")
        assert ".." not in sanitize_filename("../../etc/passwd")

    def test_sanitize_never_empty(self):
        assert sanitize_filename("") == "untitled"
        assert sanitize_filename("...") == "untitled"

    def test_slugify(self):
        assert slugify("RideCompare CatBoost!") == "ridecompare-catboost"

    def test_safe_join_ok(self, tmp_path):
        p = safe_join(tmp_path, "01_Projects/note.md")
        assert str(p).startswith(str(tmp_path.resolve()))

    def test_safe_join_rejects_escape(self, tmp_path):
        import pytest
        with pytest.raises(ValueError):
            safe_join(tmp_path, "../../../etc/passwd")

    def test_iso_roundtrip(self):
        epoch = 1_750_000_000.0
        iso = to_iso(epoch)
        assert iso.endswith("Z")
        assert abs(from_iso(iso) - epoch) < 2

    def test_from_iso_accepts_epoch_string(self):
        assert from_iso("1750000000") == 1_750_000_000.0

    def test_from_iso_garbage_returns_zero(self):
        assert from_iso("not a date") == 0.0

    def test_redact_email(self):
        clean, n = redact("contact me at jane@example.com please")
        assert n == 1
        assert "jane@example.com" not in clean

    def test_redact_assignment_keeps_key(self):
        clean, n = redact("password: hunter2secret")
        assert n >= 1
        assert "[REDACTED]" in clean

    def test_redact_clean_text_untouched(self):
        clean, n = redact("just a normal sentence about code")
        assert n == 0
        assert clean == "just a normal sentence about code"
