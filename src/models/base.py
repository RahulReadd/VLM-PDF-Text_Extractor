"""Abstract base class for all VLM adapters.

Every model family (Qwen3-VL, Qwen2.5-VL, InternVL, etc.) implements this
interface so the rest of the pipeline never cares which model is running.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image


@dataclass
class ModelConfig:
    """Hardware-aware configuration for a VLM."""

    model_id: str
    family: str
    dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    # Vision encoder resolution limits — the #1 lever for VRAM on T4.
    # Defaults tuned for 15 GB T4: keep max_pixels conservative.
    min_pixels: int = 256 * 28 * 28      # ~200 K
    max_pixels: int = 512 * 28 * 28      # ~401 K  (default 1280*28*28 causes OOM)
    quantization: str | None = None       # "4bit", "8bit", or None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run."""

    model_id: str
    vram_allocated_mb: float
    vram_reserved_mb: float
    load_time_s: float
    inference_time_s: float
    output_text: str
    output_json: dict | None  # parsed JSON if the model produced valid JSON
    json_valid: bool


class VLMAdapter(ABC):
    """Uniform interface every model adapter must implement."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        """Download (if needed) and load model + processor into GPU."""

    @abstractmethod
    def run_inference(self, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
        """Send one image + text prompt, return the raw text output."""

    def unload(self) -> None:
        """Free GPU memory so the next model can load."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_vram_usage(self) -> tuple[float, float]:
        """Return (allocated_MB, reserved_MB) on the current CUDA device."""
        if not torch.cuda.is_available():
            return (0.0, 0.0)
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        return (round(allocated, 1), round(reserved, 1))
