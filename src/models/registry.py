"""Model registry — single place to add/remove/swap VLMs.

Usage:
    from src.models import get_model
    model = get_model("qwen3-vl-4b")         # returns a loaded VLMAdapter
    model = get_model("internvl35-8b-4bit")  # quantized variant
"""

from __future__ import annotations

import torch

from .base import ModelConfig, VLMAdapter
from .qwen3_vl import Qwen3VLAdapter
from .internvl import InternVLAdapter
from .florence2 import Florence2Adapter
from .pixtral import PixtralAdapter
from .llama_vision import LlamaVisionAdapter

# ---------------------------------------------------------------------------
# Preset configurations for models that fit on a free-tier T4 (15 GB VRAM).
# To add a new model: add an entry here + an adapter class if it's a new family.
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[type[VLMAdapter], ModelConfig]] = {

    # ── 1. Qwen3-VL family (Alibaba, 2025) ───────────────────────────
    #    Best zero-shot document extraction; 30+ language OCR engine.
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

    # ── 2. InternVL 3.5 family (OpenGVLab, late 2025) ────────────────
    #    SOTA OCR among open-weights; robust structured extraction.
    "internvl35-8b-4bit": (
        InternVLAdapter,
        ModelConfig(
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
            dtype=torch.float16,
            quantization="4bit",
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ),
    ),

    # ── 3. Florence-2 (Microsoft) ─────────────────────────────────────
    #    Tiny 0.77B encoder-decoder; purpose-built for OCR + region tasks.
    #    Runs in ~2-3 GB VRAM — leaves room for batch processing.
    "florence2-large": (
        Florence2Adapter,
        ModelConfig(
            model_id="microsoft/Florence-2-large",
            family="florence2",
            dtype=torch.float16,
        ),
    ),

    # ── 4. Pixtral-12B (Mistral, 2024) ───────────────────────────────
    #    Native-resolution image processing; great on long/narrow receipts.
    #    Must run in 4-bit (~8.5 GB VRAM).
    "pixtral-12b-4bit": (
        PixtralAdapter,
        ModelConfig(
            model_id="mistral-community/pixtral-12b",
            family="pixtral",
            dtype=torch.float16,
            quantization="4bit",
        ),
    ),

    # ── 5. Llama-3.2 Vision (Meta, 2024) ─────────────────────────────
    #    Strong reasoning engine; large context window.
    #    Must run in 4-bit (~8 GB VRAM).
    #    NOTE: Requires accepting the Llama 3.2 Community License on HF.
    "llama32-11b-4bit": (
        LlamaVisionAdapter,
        ModelConfig(
            model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
            family="llama-vision",
            dtype=torch.float16,
            quantization="4bit",
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
