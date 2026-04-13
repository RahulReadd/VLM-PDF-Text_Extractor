"""Module 3: Evaluation & Scoring.

Evaluates VLM extraction outputs against ground truth:
  - Receipt fields: Exact Match, Token F1, Menu Item F1 (CORD dataset)
  - Signature detection: Accuracy, Precision, Recall, F1 (binary classification)

Usage:
    python -m app.evaluate --predictions results/outputs/preds.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def normalize_str(s: str | None) -> str:
    """Normalize a string for comparison."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[,.]$', '', s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_str(pred) == normalize_str(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and gold strings."""
    pred_tokens = set(normalize_str(pred).split())
    gold_tokens = set(normalize_str(gold).split())

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_cord_ground_truth(gt_raw) -> dict:
    """Convert CORD nested ground truth into a flat comparable dict.

    CORD ground_truth can be double-encoded (JSON string inside JSON string)
    and sub-keys like 'sub_total' / 'total' may be wrapped in a list.
    """
    gt = gt_raw
    # Unwrap string encoding (possibly double-encoded)
    while isinstance(gt, str):
        gt = json.loads(gt)

    if "gt_parse" in gt:
        gt = gt["gt_parse"]

    result = {
        "menu_items": [],
        "subtotal_price": None,
        "tax_price": None,
        "discount_price": None,
        "total_price": None,
        "cashprice": None,
        "changeprice": None,
    }

    def _collect_menu_items(menu_node, out: list):
        """Recursively extract menu items from CORD's variable menu structure.

        menu_node can be:
          - a list of dicts  →  [{"nm": ..., "price": ...}, ...]
          - a single dict    →  {"nm": ..., "price": ..., "sub": {...}}
          - a list of lists  →  rare, but handled
        Nested 'sub' items are flattened into the same list.
        """
        if isinstance(menu_node, dict):
            if "nm" in menu_node or "price" in menu_node:
                out.append({
                    "nm": menu_node.get("nm", None),
                    "cnt": menu_node.get("cnt", None),
                    "price": menu_node.get("price", None),
                })
            if "sub" in menu_node:
                _collect_menu_items(menu_node["sub"], out)
        elif isinstance(menu_node, list):
            for item in menu_node:
                _collect_menu_items(item, out)

    if "menu" in gt:
        _collect_menu_items(gt["menu"], result["menu_items"])

    # CORD sometimes wraps sub_total / total in a single-element list
    st = gt.get("sub_total")
    if isinstance(st, list):
        st = st[0] if st else {}
    if isinstance(st, dict):
        result["subtotal_price"] = st.get("subtotal_price", None)
        result["tax_price"] = st.get("tax_price", None)
        result["discount_price"] = st.get("discount_price", None)

    t = gt.get("total")
    if isinstance(t, list):
        t = t[0] if t else {}
    if isinstance(t, dict):
        result["total_price"] = t.get("total_price", None)
        result["cashprice"] = t.get("cashprice", None)
        result["changeprice"] = t.get("changeprice", None)

    return result


def evaluate_single(pred_json: dict | None, gt_parsed: dict) -> dict:
    """Compare a single prediction against parsed ground truth."""
    if pred_json is None:
        return {"json_valid": False, "field_em": 0.0, "field_f1": 0.0, "menu_f1": 0.0}

    scores = {"json_valid": True}

    scalar_fields = ["total_price", "subtotal_price", "tax_price"]
    em_scores = []
    f1_scores = []

    for field in scalar_fields:
        gold_val = gt_parsed.get(field)
        if gold_val is None:
            continue

        pred_val = None
        if "total" in pred_json and isinstance(pred_json["total"], dict):
            pred_val = pred_json["total"].get(field, pred_json["total"].get("total_price"))
        if pred_val is None and "sub_total" in pred_json and isinstance(pred_json["sub_total"], dict):
            pred_val = pred_json["sub_total"].get(field)
        if pred_val is None:
            pred_val = pred_json.get(field)

        em_scores.append(exact_match(str(pred_val), str(gold_val)))
        f1_scores.append(token_f1(str(pred_val), str(gold_val)))

    scores["field_em"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
    scores["field_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    gold_items = [normalize_str(item.get("nm", "")) for item in gt_parsed.get("menu_items", [])]
    pred_items = []
    if "menu" in pred_json and isinstance(pred_json["menu"], list):
        pred_items = [normalize_str(item.get("nm", "")) for item in pred_json["menu"]]

    if gold_items:
        gold_set = set(gold_items)
        pred_set = set(pred_items)
        common = gold_set & pred_set
        p = len(common) / len(pred_set) if pred_set else 0
        r = len(common) / len(gold_set) if gold_set else 0
        scores["menu_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        scores["menu_f1"] = 0.0

    return scores


def evaluate_batch(predictions: list[dict | None], ground_truths: list) -> dict:
    """Evaluate a batch of predictions against ground truths."""
    all_scores = []
    for pred, gt_raw in zip(predictions, ground_truths):
        gt_parsed = parse_cord_ground_truth(gt_raw)
        scores = evaluate_single(pred, gt_parsed)
        all_scores.append(scores)

    n = len(all_scores)
    if n == 0:
        return {"n": 0}

    return {
        "n": n,
        "json_valid_rate": sum(s["json_valid"] for s in all_scores) / n,
        "avg_field_em": sum(s["field_em"] for s in all_scores) / n,
        "avg_field_f1": sum(s["field_f1"] for s in all_scores) / n,
        "avg_menu_f1": sum(s["menu_f1"] for s in all_scores) / n,
        "per_sample": all_scores,
    }


# ── Signature Detection Evaluation ───────────────────────────────────────

def parse_signature_prediction(pred_json: dict | None) -> bool | None:
    """Extract the predicted signature presence from VLM output.

    Handles both unified prompt format (nested under 'signature') and
    single-task format (top-level 'signature_present').

    Returns True/False, or None if prediction couldn't be parsed.
    """
    if pred_json is None:
        return None

    # Unified prompt: {"signature": {"present": true/false, ...}}
    sig = pred_json.get("signature")
    if isinstance(sig, dict):
        val = sig.get("present", sig.get("signature_present"))
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1")

    # Single-task prompt: {"signature_present": true/false}
    val = pred_json.get("signature_present")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")

    return None


def evaluate_signature_single(pred_json: dict | None, gt_present: bool) -> dict:
    """Evaluate a single signature detection prediction.

    Args:
        pred_json: Parsed JSON from VLM output.
        gt_present: Ground truth — True if signature is present.

    Returns:
        Dict with 'pred', 'gold', 'correct', 'tp', 'fp', 'fn', 'tn'.
    """
    pred = parse_signature_prediction(pred_json)

    if pred is None:
        pred = False  # conservative: treat unparseable as "no signature"

    return {
        "pred": pred,
        "gold": gt_present,
        "correct": pred == gt_present,
        "tp": int(pred and gt_present),
        "fp": int(pred and not gt_present),
        "fn": int(not pred and gt_present),
        "tn": int(not pred and not gt_present),
    }


def evaluate_signature_batch(predictions: list[dict | None], ground_truths: list[bool]) -> dict:
    """Evaluate a batch of signature detection predictions.

    Args:
        predictions: List of parsed JSON outputs from VLM.
        ground_truths: List of booleans — True if signature is present.

    Returns:
        Dict with accuracy, precision, recall, f1, and per-sample results.
    """
    all_scores = []
    for pred, gt in zip(predictions, ground_truths):
        scores = evaluate_signature_single(pred, gt)
        all_scores.append(scores)

    n = len(all_scores)
    if n == 0:
        return {"n": 0}

    tp = sum(s["tp"] for s in all_scores)
    fp = sum(s["fp"] for s in all_scores)
    fn = sum(s["fn"] for s in all_scores)
    tn = sum(s["tn"] for s in all_scores)

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n": n,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "per_sample": all_scores,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VLM extraction against CORD ground truth")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--output", type=str, default="results/eval_results.json", help="Output path")
    args = parser.parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} predictions from {args.predictions}")
    preds = [d.get("parsed_json") for d in data]
    gts = [d.get("ground_truth") for d in data]

    results = evaluate_batch(preds, gts)

    print(f"\nJSON valid rate:  {results['json_valid_rate']:.1%}")
    print(f"Field Exact Match:{results['avg_field_em']:.3f}")
    print(f"Field F1:         {results['avg_field_f1']:.3f}")
    print(f"Menu F1:          {results['avg_menu_f1']:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")
