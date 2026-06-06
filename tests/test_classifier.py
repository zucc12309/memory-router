"""Tests for the rule-based classifier."""

from memory_router.classifier import classify, Classification


def test_code_task_detection():
    cls = classify("write a function to sort a list in python")
    assert cls.task == "code"
    assert cls.domain == "software"


def test_code_task_detection_for_pytest_module_prompt():
    cls = classify("Write pytest tests for the auth module")
    assert cls.task == "code"
    assert cls.domain == "software"


def test_security_task_detection():
    cls = classify("check for SQL injection vulnerabilities in this endpoint")
    assert cls.task == "security"


def test_reasoning_task_detection():
    cls = classify("prove that the sum of angles in a triangle is 180 degrees")
    assert cls.task == "reasoning"


def test_explain_task_detection():
    cls = classify("explain how HTTP cookies work")
    assert cls.task == "explain"


def test_summarize_task_detection():
    cls = classify("summarize the key points from this meeting")
    assert cls.task == "summarize"


def test_finance_domain_detection():
    cls = classify("what is the portfolio yield of this bond fund")
    assert cls.domain == "finance"


def test_software_domain_detection_for_stack_questions():
    cls = classify("Which stack does the project use?")
    assert cls.domain == "software"


def test_ml_domain_detection():
    cls = classify("train a neural network with pytorch for image classification")
    assert cls.domain == "ml"


def test_general_fallback():
    cls = classify("hello how are you today")
    assert cls.task == "general"
    assert cls.domain == "general"


def test_concepts_extracted():
    cls = classify("implement authentication using JWT tokens in typescript")
    assert len(cls.concepts) > 0
    assert any("jwt" in c or "typescript" in c or "authentication" in c for c in cls.concepts)


def test_complexity_increases_with_length():
    short = classify("hi")
    long = classify(
        "design a distributed microservices architecture with event sourcing, "
        "CQRS, and a threat model for the authentication layer, "
        "optimizing for horizontal scalability across multiple regions"
    )
    assert long.complexity > short.complexity


def test_complexity_caps_at_one():
    cls = classify("a" * 5000)
    assert cls.complexity <= 1.0


def test_to_dict():
    cls = classify("write python code")
    d = cls.to_dict()
    assert "task" in d
    assert "domain" in d
    assert "concepts" in d
    assert "complexity" in d


def test_empty_query():
    cls = classify("")
    assert isinstance(cls, Classification)
    assert cls.task == "general"
