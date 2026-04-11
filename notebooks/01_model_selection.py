"""
Phase 0: Model Selection Benchmark
====================================
Run this in Google Colab (Runtime → Change runtime type → T4 GPU).

Copy-paste cells into Colab, or upload the whole repo and run sections.
Each section is delimited by # %% (Jupyter cell marker).
"""

# %% [markdown]
# # Phase 0: VLM Model Selection Benchmark
#
# **Goal:** Test candidate VLMs on CORD receipt images and pick the best one for our pipeline.
#
# **What we measure:**
# - VRAM usage after loading (does it fit on T4?)
# - Inference time per image
# - JSON compliance (can we parse the output?)
# - Output quality (visual inspection)

# %% — Cell 1: Install dependencies
# !pip install -q git+https://github.com/huggingface/transformers
# !pip install -q accelerate bitsandbytes qwen-vl-utils[decord] datasets pillow

# If running from the repo root:
# !git clone https://github.com/YOUR_USERNAME/VLM_PDF_Extractor.git
# %cd VLM_PDF_Extractor

# %% — Cell 2: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**2:.0f} MB")

# %% — Cell 3: Load sample images from CORD dataset
from datasets import load_dataset
from PIL import Image

cord = load_dataset("naver-clova-ix/cord-v2", split="test")

# Pick 3 diverse receipt images for benchmarking
sample_indices = [0, 10, 25]
test_images: list[Image.Image] = []
test_ground_truths: list[dict] = []

for idx in sample_indices:
    sample = cord[idx]
    img = sample["image"]
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    test_images.append(img.convert("RGB"))
    test_ground_truths.append(sample["ground_truth"])
    print(f"Image {idx}: {img.size}")

print(f"\nLoaded {len(test_images)} test images from CORD dataset.")

# %% — Cell 4: Define the benchmark prompt
# This is the prompt we'll use for ALL models — structured JSON extraction for receipts.
BENCHMARK_PROMPT = """You are a document extraction assistant. Analyze this receipt image and extract all information as a JSON object with this exact structure:

{
  "store_name": "...",
  "date": "...",
  "items": [
    {"name": "...", "quantity": "...", "price": "..."}
  ],
  "subtotal": "...",
  "tax": "...",
  "total": "...",
  "payment_method": "..."
}

Rules:
- Return ONLY the JSON object, no extra text.
- Use null for fields you cannot find.
- Extract ALL line items visible in the receipt.
"""

# %% — Cell 5: Run the benchmark
import sys
sys.path.insert(0, ".")  # ensure src/ is importable

from src.benchmark import run_benchmark, print_report, save_report_csv

# Models to compare — comment out any that don't fit or take too long
MODELS_TO_TEST = [
    "qwen3-vl-2b",       # Latest (Oct 2025), 2.1B params — should fit easily
    "qwen25-vl-3b",      # Qwen2.5, 3B params — slightly larger
    # "qwen3-vl-4b",      # 4B params — may be tight on VRAM
    # "qwen25-vl-7b-awq", # 7B 4-bit quantized — worth testing if others fit
    # "internvl25-2b",    # Different family for diversity
]

reports = run_benchmark(
    model_names=MODELS_TO_TEST,
    images=test_images,
    prompts=[BENCHMARK_PROMPT],  # same prompt for all images
    max_new_tokens=1024,
)

# %% — Cell 6: Print comparison report
print_report(reports)

# %% — Cell 7: Save results
save_report_csv(reports, "results/benchmark_summary.csv")

# Also save detailed JSON outputs per model for later analysis
import json
import os
os.makedirs("results/benchmark_outputs", exist_ok=True)

for report in reports:
    output = {
        "model_name": report.model_name,
        "model_id": report.model_id,
        "load_time_s": report.load_time_s,
        "vram_allocated_mb": report.vram_after_load_mb[0],
        "vram_reserved_mb": report.vram_after_load_mb[1],
        "avg_inference_s": report.avg_inference_s,
        "json_success_rate": report.json_success_rate,
        "per_image": [
            {
                "inference_time_s": r.inference_time_s,
                "json_valid": r.json_valid,
                "output_text": r.output_text,
                "output_json": r.output_json,
            }
            for r in report.per_image_results
        ],
    }
    path = f"results/benchmark_outputs/{report.model_name}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")

# %% — Cell 8: Visual comparison — show image alongside extracted JSON
import matplotlib.pyplot as plt

for report in reports:
    fig, axes = plt.subplots(1, len(test_images), figsize=(6 * len(test_images), 6))
    if len(test_images) == 1:
        axes = [axes]
    fig.suptitle(f"Model: {report.model_name}", fontsize=14, fontweight="bold")

    for i, (img, res) in enumerate(zip(test_images, report.per_image_results)):
        axes[i].imshow(img)
        axes[i].set_title(
            f"Img {i+1} | {res.inference_time_s}s | JSON: {'Y' if res.json_valid else 'N'}",
            fontsize=10,
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"results/benchmark_outputs/{report.model_name}_visual.png", dpi=100, bbox_inches="tight")
    plt.show()

# %% — Cell 9: Compare against CORD ground truth (quick sanity check)
import json as _json

print("\n" + "=" * 60)
print("  GROUND TRUTH vs MODEL OUTPUT (first image)")
print("=" * 60)

gt_raw = test_ground_truths[0]
if isinstance(gt_raw, str):
    gt_parsed = _json.loads(gt_raw)
else:
    gt_parsed = gt_raw

print(f"\nGround Truth Keys: {list(gt_parsed.keys()) if isinstance(gt_parsed, dict) else type(gt_parsed)}")
print(f"Ground Truth (preview): {str(gt_parsed)[:300]}")

for report in reports:
    res = report.per_image_results[0]
    print(f"\n--- {report.model_name} ---")
    if res.output_json:
        print(f"Extracted Keys: {list(res.output_json.keys())}")
        print(f"Extracted (preview): {str(res.output_json)[:300]}")
    else:
        print(f"Raw output (not valid JSON): {res.output_text[:300]}")

# %% [markdown]
# ## Decision Criteria
#
# | Criteria | Weight | Notes |
# |----------|--------|-------|
# | VRAM fits T4 | Must-have | Must load + infer within 15 GB |
# | JSON compliance | High | Must produce parseable JSON consistently |
# | Inference speed | Medium | <10s per image preferred for batch processing |
# | Extraction accuracy | High | Compare against CORD ground truth |
# | Model recency | Low | Newer models preferred but not at the cost of stability |
#
# **Pick the model that scores best across all criteria and use it as the
# primary model for the pipeline. Keep the runner-up for the "Compare 2 VLMs" bonus.**
