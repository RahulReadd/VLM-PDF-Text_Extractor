"""Model benchmarking framework for Phase 0 comparison.

Runs each candidate VLM on the same set of images and prompts, recording:
- VRAM usage (allocated & reserved)
- Model load time
- Per-image inference time
- Raw output text
- Whether the output is valid JSON
- Side-by-side comparison table

Usage (from Colab or CLI):
    from src.benchmark import run_benchmark, print_report
    results = run_benchmark(
        model_names=["qwen3-vl-2b", "qwen25-vl-3b"],
        images=[pil_img1, pil_img2],
        prompts=["Extract all key-value pairs as JSON.", ...],
    )
    print_report(results)
"""

from __future__ import annotations

import json
import time
import gc
from dataclasses import dataclass, field

import torch
from PIL import Image

from .models.base import BenchmarkResult
from .models.registry import ModelRegistry, get_model


@dataclass
class ModelReport:
    """Aggregated report for one model across all test images."""

    model_name: str
    model_id: str
    load_time_s: float
    vram_after_load_mb: tuple[float, float]
    per_image_results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def avg_inference_s(self) -> float:
        times = [r.inference_time_s for r in self.per_image_results]
        return sum(times) / len(times) if times else 0.0

    @property
    def json_success_rate(self) -> float:
        if not self.per_image_results:
            return 0.0
        valid = sum(1 for r in self.per_image_results if r.json_valid)
        return valid / len(self.per_image_results)


def _try_parse_json(text: str) -> tuple[dict | None, bool]:
    """Attempt to extract JSON from model output (handles markdown fences)."""
    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed, True
    except json.JSONDecodeError:
        pass
    # Try to find the first { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            return parsed, True
        except json.JSONDecodeError:
            pass
    return None, False


def benchmark_single_model(
    model_name: str,
    images: list[Image.Image],
    prompts: list[str],
    max_new_tokens: int = 1024,
) -> ModelReport:
    """Load one model, run all images x prompts, then unload."""

    config = ModelRegistry.get_config(model_name)
    print(f"\n{'='*60}")
    print(f"  Loading: {model_name} ({config.model_id})")
    print(f"  dtype={config.dtype}, quant={config.quantization}")
    print(f"{'='*60}")

    # Clear GPU before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    adapter = get_model(model_name)
    load_time = time.time() - t0
    vram_after_load = adapter.get_vram_usage()

    print(f"  Loaded in {load_time:.1f}s | VRAM: {vram_after_load[0]:.0f} MB allocated, {vram_after_load[1]:.0f} MB reserved")

    report = ModelReport(
        model_name=model_name,
        model_id=config.model_id,
        load_time_s=round(load_time, 2),
        vram_after_load_mb=vram_after_load,
    )

    for i, (img, prompt) in enumerate(zip(images, prompts)):
        print(f"  Image {i+1}/{len(images)}: ", end="", flush=True)

        try:
            t1 = time.time()
            raw_output = adapter.run_inference(img, prompt, max_new_tokens=max_new_tokens)
            infer_time = time.time() - t1

            vram_now = adapter.get_vram_usage()
            parsed, is_valid = _try_parse_json(raw_output)

            result = BenchmarkResult(
                model_id=config.model_id,
                vram_allocated_mb=vram_now[0],
                vram_reserved_mb=vram_now[1],
                load_time_s=report.load_time_s,
                inference_time_s=round(infer_time, 2),
                output_text=raw_output,
                output_json=parsed,
                json_valid=is_valid,
            )
            report.per_image_results.append(result)
            print(f"{infer_time:.1f}s | JSON valid: {is_valid} | VRAM: {vram_now[0]:.0f} MB")

        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA ERROR — skipping remaining images for {model_name}")
                print(f"  Error: {e}")
                break
            print(f"ERROR — {e} (skipping image)")
            result = BenchmarkResult(
                model_id=config.model_id,
                vram_allocated_mb=0, vram_reserved_mb=0,
                load_time_s=report.load_time_s,
                inference_time_s=0, output_text=f"ERROR: {e}",
                output_json=None, json_valid=False,
            )
            report.per_image_results.append(result)

    # Unload to free VRAM for the next model
    adapter.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report


def run_benchmark(
    model_names: list[str],
    images: list[Image.Image],
    prompts: list[str],
    max_new_tokens: int = 1024,
) -> list[ModelReport]:
    """Run the full benchmark across all models, sequentially."""

    if len(images) != len(prompts):
        # If a single prompt is given, replicate for all images
        if len(prompts) == 1:
            prompts = prompts * len(images)
        else:
            raise ValueError(
                f"images ({len(images)}) and prompts ({len(prompts)}) must have same length, "
                "or provide a single prompt to use for all images."
            )

    reports = []
    for name in model_names:
        try:
            report = benchmark_single_model(name, images, prompts, max_new_tokens)
            reports.append(report)
        except Exception as e:
            print(f"\n  FAILED: {name} — {e}")
            # Still try to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return reports


def print_report(reports: list[ModelReport]) -> None:
    """Print a comparison table to stdout."""

    print(f"\n{'='*80}")
    print("  MODEL COMPARISON REPORT")
    print(f"{'='*80}\n")

    header = f"{'Model':<25} {'Load(s)':<9} {'VRAM(MB)':<12} {'Avg Inf(s)':<12} {'JSON %':<10}"
    print(header)
    print("-" * len(header))

    for r in reports:
        vram_str = f"{r.vram_after_load_mb[0]:.0f}/{r.vram_after_load_mb[1]:.0f}"
        print(
            f"{r.model_name:<25} "
            f"{r.load_time_s:<9.1f} "
            f"{vram_str:<12} "
            f"{r.avg_inference_s:<12.1f} "
            f"{r.json_success_rate*100:<10.0f}"
        )

    # Detailed per-image outputs
    print(f"\n{'='*80}")
    print("  DETAILED OUTPUTS")
    print(f"{'='*80}")

    for r in reports:
        print(f"\n--- {r.model_name} ---")
        for i, res in enumerate(r.per_image_results):
            print(f"\n  [Image {i+1}] ({res.inference_time_s}s, JSON valid: {res.json_valid})")
            # Truncate very long outputs for display
            preview = res.output_text[:500]
            if len(res.output_text) > 500:
                preview += "... [truncated]"
            print(f"  {preview}")


def save_report_csv(reports: list[ModelReport], path: str = "results/benchmark_summary.csv") -> None:
    """Save summary as CSV for later use in the write-up."""
    import csv
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "model_id", "load_time_s",
            "vram_allocated_mb", "vram_reserved_mb",
            "avg_inference_s", "json_success_rate",
        ])
        for r in reports:
            writer.writerow([
                r.model_name,
                r.model_id,
                r.load_time_s,
                r.vram_after_load_mb[0],
                r.vram_after_load_mb[1],
                round(r.avg_inference_s, 2),
                round(r.json_success_rate, 3),
            ])
    print(f"\nSaved benchmark CSV to: {path}")
