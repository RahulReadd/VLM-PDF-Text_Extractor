"""Model registry — single place to add/remove/swap VLMs.

Usage:
    from src.models import get_model
    model = get_model("qwen3-vl-2b")   # returns a loaded VLMAdapter
    model = get_model("qwen25-vl-7b-awq")
"""

from __future__ import annotations

import torch

from .base import ModelConfig, VLMAdapter
from .qwen3_vl import Qwen3VLAdapter
from .qwen25_vl import Qwen25VLAdapter
from .internvl import InternVLAdapter

# ---------------------------------------------------------------------------
# Preset configurations for models that fit on a free-tier T4 (15 GB VRAM).
# To add a new model: add an entry here + an adapter class if it's a new family.
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[type[VLMAdapter], ModelConfig]] = {
    # ── Qwen3-VL family (latest, Oct 2025) ──────────────────────────
    "qwen3-vl-2b": (
        Qwen3VLAdapter,
        ModelConfig(
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            family="qwen3-vl",
            dtype=torch.float16,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),
    "qwen3-vl-4b": (
        Qwen3VLAdapter,
        ModelConfig(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            family="qwen3-vl",
            dtype=torch.float16,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),
    "qwen3-vl-4b-4bit": (
        Qwen3VLAdapter,
        ModelConfig(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            family="qwen3-vl",
            dtype=torch.float16,
            quantization="4bit",
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),

    # ── Qwen2.5-VL family (Jan 2025) ────────────────────────────────
    "qwen25-vl-3b": (
        Qwen25VLAdapter,
        ModelConfig(
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            family="qwen2.5-vl",
            dtype=torch.float16,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),
    "qwen25-vl-7b-awq": (
        Qwen25VLAdapter,
        ModelConfig(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            family="qwen2.5-vl",
            dtype=torch.float16,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),

    # ── InternVL family ──────────────────────────────────────────────
    "internvl25-2b": (
        InternVLAdapter,
        ModelConfig(
            model_id="OpenGVLab/InternVL2_5-2B",
            family="internvl",
            dtype=torch.float16,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),
}


class ModelRegistry:
    """Discover available presets and instantiate adapters."""

    @staticmethod
    def list_models() -> list[str]:
        return list(PRESETS.keys())

    @staticmethod
    def get_config(name: str) -> ModelConfig:
        if name not in PRESETS:
            raise KeyError(
                f"Unknown model '{name}'. Available: {list(PRESETS.keys())}"
            )
        return PRESETS[name][1]

    @staticmethod
    def create(name: str, config_overrides: dict | None = None) -> VLMAdapter:
        """Instantiate an adapter (does NOT call .load() yet)."""
        if name not in PRESETS:
            raise KeyError(
                f"Unknown model '{name}'. Available: {list(PRESETS.keys())}"
            )
        adapter_cls, default_config = PRESETS[name]

        if config_overrides:
            from dataclasses import replace
            config = replace(default_config, **config_overrides)
        else:
            config = default_config

        return adapter_cls(config)


def get_model(name: str, config_overrides: dict | None = None) -> VLMAdapter:
    """Convenience: create and load in one call."""
    adapter = ModelRegistry.create(name, config_overrides)
    adapter.load()
    return adapter
