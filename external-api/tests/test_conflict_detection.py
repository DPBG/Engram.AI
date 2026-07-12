"""Unit tests for knowledge conflict detection helpers."""

import asyncio
import importlib.util
import os
import sys
from unittest.mock import AsyncMock

_CD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "external_api", "conflict_detection.py"
)
_spec = importlib.util.spec_from_file_location("external_conflict_detection", _CD_PATH)
cd = importlib.util.module_from_spec(_spec)
sys.modules["external_conflict_detection"] = cd
_spec.loader.exec_module(cd)

cosine_similarity = cd.cosine_similarity
detect_knowledge_conflict = cd.detect_knowledge_conflict
word_overlap_conflict = cd.word_overlap_conflict


def _run(coro):
    return asyncio.run(coro)


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_word_overlap_conflict_flags_unrelated_text():
    assert word_overlap_conflict("alpha beta gamma", "zeta eta theta") is True


def test_word_overlap_conflict_accepts_identical_text():
    assert word_overlap_conflict("the quick brown fox", "the quick brown fox") is False


def test_detect_knowledge_conflict_semantic_paraphrase_not_conflict():
    embeddings = AsyncMock()
    embeddings.embed_text.side_effect = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
    ]
    assert (
        _run(
            detect_knowledge_conflict(
                "A feline rests on the mat",
                {"description": "The cat sits on the mat"},
                embedding_service=embeddings,
            )
        )
        is False
    )


def test_detect_knowledge_conflict_semantic_contradiction():
    embeddings = AsyncMock()
    embeddings.embed_text.side_effect = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert (
        _run(
            detect_knowledge_conflict(
                "The sky is green",
                {"description": "The sky is blue"},
                embedding_service=embeddings,
            )
        )
        is True
    )


def test_detect_knowledge_conflict_falls_back_on_embedding_failure():
    embeddings = AsyncMock()
    embeddings.embed_text.side_effect = RuntimeError("ollama down")
    assert (
        _run(
            detect_knowledge_conflict(
                "alpha beta gamma",
                {"description": "zeta eta theta iota"},
                embedding_service=embeddings,
            )
        )
        is True
    )
