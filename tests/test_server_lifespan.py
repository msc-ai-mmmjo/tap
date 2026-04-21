"""Tests for the FastAPI lifespan model-load guard.

Modal's @modal.enter() preloads Hydra into server._models before
FastAPI's startup runs. The lifespan must skip load_hydra when the
entry already exists, while still loading BERT (which Modal doesn't
preload).
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from fastapi import FastAPI  # noqa: E402

from app.backend import server  # noqa: E402


def _run_lifespan() -> None:
    async def _run() -> None:
        async with server.lifespan(FastAPI()):
            pass

    asyncio.run(_run())


def test_lifespan_skips_hydra_when_already_preloaded() -> None:
    preloaded_hydra = MagicMock(name="preloaded_hydra")
    preloaded_hydra_tok = MagicMock(name="preloaded_hydra_tok")
    bert_model = MagicMock(name="bert")
    bert_tok = MagicMock(name="bert_tok")

    with (
        patch.dict(server._models, {"hydra": preloaded_hydra}, clear=True),
        patch.dict(server._tokenizers, {"hydra": preloaded_hydra_tok}, clear=True),
        patch.object(server, "load_hydra") as mock_load_hydra,
        patch.object(
            server, "load_bert", return_value=(bert_model, bert_tok)
        ) as mock_load_bert,
    ):
        _run_lifespan()

        mock_load_hydra.assert_not_called()
        mock_load_bert.assert_called_once()


def test_lifespan_loads_both_when_nothing_preloaded() -> None:
    hydra_model = MagicMock(name="hydra")
    hydra_tok = MagicMock(name="hydra_tok")
    bert_model = MagicMock(name="bert")
    bert_tok = MagicMock(name="bert_tok")

    with (
        patch.dict(server._models, {}, clear=True),
        patch.dict(server._tokenizers, {}, clear=True),
        patch.object(
            server, "load_hydra", return_value=(hydra_model, hydra_tok)
        ) as mock_load_hydra,
        patch.object(
            server, "load_bert", return_value=(bert_model, bert_tok)
        ) as mock_load_bert,
    ):
        _run_lifespan()

        mock_load_hydra.assert_called_once()
        mock_load_bert.assert_called_once()
