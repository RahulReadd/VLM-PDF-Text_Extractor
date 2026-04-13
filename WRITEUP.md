# VLM-Based PDF Document Extraction — Technical Write-Up

## 1. VLM Selection: What We Used and What We Considered

**Selected model: Qwen3-VL-2B-Instruct** (Alibaba, 2025)

We benchmarked 7 open-source VLMs on 20 CORD v2 receipt images using a free Google Colab T4 GPU (15 GB VRAM). All models were evaluated on operational metrics (load time, VRAM, inference speed, JSON parse rate) and accuracy metrics (Field Exact Match, Field Token F1, Menu Item F1).

| Model | Params | VRAM (MB) | Avg Inference (s) | JSON % | Notes |
|-------|--------|-----------|--------------------|---------|-|
| **Qwen3-VL-2B** | 2B | 4,058 | 8.41 | 100% | **Selected** — fastest load, lowest VRAM |
| Qwen2.5-VL-3B | 3B | 7,171 | 8.98 | 100% | Strong but 1.8x the VRAM |
| Qwen3-VL-4B | 4B | 8,465 | 10.43 | 100% | Diminishing returns over 2B |
| InternVL 3.5-8B (4-bit) | 8B | 6,307 | 14.62 | 100% | SOTA OCR but slow load (~22 min) |
| Pixtral-12B (4-bit) | 12B | 8,693 | 22.82 | 100% | Native-resolution, but 2.7x slower inference |
| Florence-2-large | 0.77B | ~2,000 | — | — | Incompatible prompt architecture; uses task tokens (`<OCR>`), not free-form prompts |
| Llama-3.2-11B-Vision (4-bit) | 11B | ~8,000 | — | — | Gated model; CUDA OOM under load on T4 |

**Why Qwen3-VL-2B?** It loads 4x faster than the runner-up, uses half the VRAM (leaving 11 GB headroom for high-resolution documents), and achieves the fastest inference while maintaining 100% structured JSON output. On a free T4 with tight session limits, the low resource footprint is decisive — it allows iterative experimentation without constant OOM crashes.

**Why not the others?** InternVL and Pixtral offer stronger raw OCR scores on academic benchmarks, but their 4-bit quantized inference is 1.7–2.7x slower, and InternVL's 22-minute load time makes interactive/demo use impractical. Florence-2 is architecturally incompatible with our unified prompt strategy (it only accepts predefined task tokens). Llama-3.2-11B requires a gated license and consistently hits CUDA memory limits on T4 when processing full-resolution document images.

## 2. Prompting Strategy

**Approach: Single unified prompt returning all four extraction types in one JSON object.**

Rather than making 4 separate VLM calls per page (one for key-value, one for signature, one for form fields, one for receipt), we designed a single structured prompt that asks the model to return everything at once:

```json
{
  "key_value_pairs": { ... },
  "signature": { "present": true/false, "confidence": "...", "location": "..." },
  "form_fields": [ { "field_name": "...", "status": "filled/empty", "value": "..." } ],
  "receipt": { "menu": [...], "sub_total": {...}, "total": {...} }
}
```

**Why this works best:**

1. **Efficiency** — One inference pass per page instead of four. At ~8.4s per inference on T4, this saves ~25s per page.
2. **Context sharing** — The model sees the full document once and can cross-reference information (e.g., recognizing a receipt total near form fields, or a signature below a key-value section).
3. **JSON schema as prompt** — By embedding the exact JSON structure the model should produce, with descriptive field names and example values, we observed 100% valid JSON output. The model treats the schema as a template and fills it in.
4. **Null convention** — Instructing the model to use `null` for missing fields (e.g., `"receipt": null` for non-receipt documents) prevents hallucination of nonexistent data.
5. **Fallback parsing** — The `parse_json_output` function handles edge cases: markdown-fenced responses, leading text before JSON, and nested brace extraction. This makes the pipeline robust even when the model adds minor formatting artifacts.

We retain individual task-specific prompts as a fallback for benchmarking and ablation studies, but the unified prompt is the production default.

## 3. Failure Cases and Limitations

**Observed failure modes:**

- **Dense multi-column receipts:** When receipts have >15 line items in small print, the model occasionally merges adjacent items or misaligns prices with item names. The 401K pixel budget forces downscaling of large pages, which blurs fine text.
- **Handwritten text:** The model reliably detects *whether* a signature is present (binary classification) but struggles to transcribe handwritten annotations or cursive text adjacent to signatures.
- **Rotated sub-regions:** Our deskew pipeline corrects whole-page skew via Hough line detection, but a rotated table embedded in an otherwise straight page is not detected — the VLM must handle it, and it often fails on severely rotated (>15°) sub-regions.
- **Non-Latin scripts:** While Qwen3-VL supports 30+ languages, our evaluation was limited to English/Indonesian (CORD dataset). Accuracy on Arabic, CJK, or mixed-script documents is untested.
- **Florence-2 / Llama incompatibility:** Two of seven candidate models couldn't complete benchmarking — Florence-2 due to its task-token architecture (not free-form prompt compatible), and Llama-3.2-11B due to gating + VRAM limits. This reduced our model selection pool.
- **CUDA context corruption:** A device-side assert in one model permanently corrupts the CUDA context for the entire Colab session, requiring a full runtime restart. We mitigated this by running each model in its own cell and saving results to disk/Drive for cross-session recovery.

## 4. Production Improvements

If deploying this pipeline at scale, the key improvements would be:

1. **Dedicated GPU infrastructure:** Replace Colab's ephemeral T4 with a persistent A100/H100 instance. This enables larger models (InternVL-8B full precision, Qwen3-VL-7B) that scored higher on OCR benchmarks, and eliminates the VRAM gymnastics.

2. **vLLM / TGI serving:** Use a production inference server (vLLM, HuggingFace TGI) for batched, paged-attention inference. Current sequential processing (~8.4s/page) would drop to ~1-2s/page with proper batching and KV-cache management.

3. **Fine-tuning on domain data:** The current pipeline is fully zero-shot. Fine-tuning Qwen3-VL on 500–1000 labeled examples from the target document domain (invoices, tax forms, medical records) would significantly improve field extraction accuracy, especially for domain-specific schemas.

4. **OCR pre-pass + VLM verification:** For documents with very fine print, run a dedicated OCR engine (Tesseract, PaddleOCR) first, then feed the OCR text alongside the image to the VLM. This two-stage approach provides the VLM with text it might miss at lower resolutions.

5. **Region-level skew correction:** Replace the current global deskew with a layout-analysis step (e.g., LayoutLMv3 or DocTR) that detects and individually corrects rotated tables, stamps, or annotations within a page.

6. **Confidence scoring and human-in-the-loop:** Add confidence thresholds to flag low-confidence extractions for human review. The unified prompt already returns a `confidence` field for signatures; extend this pattern to all extraction types.

7. **Caching and incremental processing:** For document re-processing (e.g., updated PDFs), cache page hashes and only re-extract changed pages. Store results in a structured database rather than flat JSON files.

8. **Evaluation at scale:** Expand beyond CORD (receipts) and SigDetectVerifyFlow (signatures) to include FUNSD (form understanding), SROIE (scanned receipts), and DocVQA (document visual QA) benchmarks for a more comprehensive accuracy picture.
