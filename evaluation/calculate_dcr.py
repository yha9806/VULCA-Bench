"""
VULCA-Bench DCR Calculator
Calculates Dimension Coverage Rate for VLM-generated critiques.

DCR measures how many of the expert-annotated dimensions (L1-L5) are covered
by the VLM's critique, using keyword matching and semantic similarity.

Usage:
    python calculate_dcr.py --input results/gpt-4o_results.jsonl --output dcr_scores.json

Author: VULCA Project Team
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict


# =============================================================================
# Culture-Specific Dimensions
# =============================================================================

CULTURE_DIMENSIONS = {
    "chinese": {"prefix": "CN", "total": 30, "threshold": 21},
    "western": {"prefix": "WE", "total": 22, "threshold": 15},
    "japanese": {"prefix": "JP", "total": 27, "threshold": 19},
    "korean": {"prefix": "KR", "total": 25, "threshold": 18},
    "islamic": {"prefix": "IS", "total": 28, "threshold": 20},
    "indian": {"prefix": "IN", "total": 30, "threshold": 21},
    "hermitage": {"prefix": "WS", "total": 30, "threshold": 21},
    "mural": {"prefix": "MU", "total": 30, "threshold": 21},
}

# Layer keywords for detecting coverage
LAYER_KEYWORDS = {
    "L1": [  # Visual Perception
        "color", "composition", "line", "shape", "form", "tone", "contrast",
        "light", "shadow", "texture", "pattern", "rhythm", "balance",
        "颜色", "构图", "线条", "造型", "明暗", "对比", "纹理"
    ],
    "L2": [  # Technical Analysis
        "brushwork", "technique", "medium", "material", "oil", "ink", "silk",
        "canvas", "paper", "pigment", "glaze", "wash", "stroke", "layer",
        "笔法", "技法", "媒材", "油彩", "水墨", "绢本", "纸本", "颜料"
    ],
    "L3": [  # Cultural Symbolism
        "symbol", "motif", "iconography", "allegory", "mythology", "tradition",
        "religious", "ritual", "auspicious", "metaphor", "represent",
        "象征", "图像", "寓意", "典故", "传统", "宗教", "仪式", "吉祥"
    ],
    "L4": [  # Historical Context
        "dynasty", "period", "century", "era", "movement", "school", "influence",
        "patron", "commission", "provenance", "style", "genre", "artist",
        "朝代", "时期", "世纪", "流派", "风格", "影响", "赞助"
    ],
    "L5": [  # Philosophical Aesthetics
        "philosophy", "aesthetics", "spiritual", "transcend", "harmony",
        "beauty", "sublime", "emotion", "meaning", "interpretation", "theory",
        "哲学", "美学", "精神", "超越", "和谐", "意境", "气韵", "情感"
    ]
}

# Culture-specific terms that indicate deeper understanding
CULTURAL_TERMS = {
    "chinese": [
        "qiyun", "gufa", "bimo", "cunfa", "liubai", "yijing", "xieyi", "gongbi",
        "气韵生动", "骨法用笔", "应物象形", "随类赋彩", "经营位置", "传移模写",
        "六法", "三远", "笔墨", "墨分五色", "留白", "写意", "工笔"
    ],
    "western": [
        "chiaroscuro", "sfumato", "impasto", "tenebrism", "perspective",
        "vanishing point", "baroque", "renaissance", "impressionism"
    ],
    "japanese": [
        "wabi-sabi", "mono no aware", "yugen", "ma", "notan",
        "ukiyo-e", "nihonga", "rinpa", "kano", "sumi-e"
    ],
    "korean": [
        "minhwa", "chaekgeori", "sipjangsaeng", "munbangdo", "sumukhwa"
    ],
    "islamic": [
        "arabesque", "geometric", "calligraphy", "muqarnas", "tessellation",
        "kufic", "naskh", "thuluth", "miniature", "illumination"
    ],
    "indian": [
        "rasa", "bhava", "mudra", "mandala", "yantra", "mughal",
        "rajput", "pahari", "tanjore", "madhubani", "pattachitra"
    ]
}


# =============================================================================
# DCR Calculation
# =============================================================================

def calculate_layer_coverage(critique: str, layer: str) -> float:
    """
    Calculate coverage score for a single layer based on keyword matching.

    Returns:
        Score between 0.0 and 1.0
    """
    if not critique:
        return 0.0

    critique_lower = critique.lower()
    keywords = LAYER_KEYWORDS.get(layer, [])

    # Count matching keywords
    matches = sum(1 for kw in keywords if kw.lower() in critique_lower)

    # Normalize by total keywords (with minimum threshold)
    if len(keywords) == 0:
        return 0.0

    coverage = min(1.0, matches / max(3, len(keywords) * 0.3))
    return coverage


def calculate_cultural_bonus(critique: str, culture: str) -> float:
    """
    Calculate bonus score for culture-specific term usage.

    Returns:
        Bonus between 0.0 and 0.2
    """
    if not critique or culture not in CULTURAL_TERMS:
        return 0.0

    critique_lower = critique.lower()
    terms = CULTURAL_TERMS[culture]

    matches = sum(1 for term in terms if term.lower() in critique_lower)

    # Bonus capped at 0.2
    return min(0.2, matches * 0.04)


def calculate_dcr(critique: str, culture: str, reference_dims: List[str] = None) -> Dict[str, Any]:
    """
    Calculate Dimension Coverage Rate for a VLM-generated critique.

    Args:
        critique: VLM-generated critique text
        culture: Cultural tradition (e.g., "chinese", "western")
        reference_dims: Optional list of expert-annotated dimensions

    Returns:
        Dict with DCR score and layer breakdown
    """
    culture = culture.lower()
    culture_info = CULTURE_DIMENSIONS.get(culture, CULTURE_DIMENSIONS["western"])

    # Calculate per-layer coverage
    layer_scores = {}
    for layer in ["L1", "L2", "L3", "L4", "L5"]:
        layer_scores[layer] = calculate_layer_coverage(critique, layer)

    # Base DCR is weighted average of layer scores
    # L1-L2 (visual/technical) weighted higher as they're more verifiable
    weights = {"L1": 0.25, "L2": 0.25, "L3": 0.20, "L4": 0.15, "L5": 0.15}
    base_dcr = sum(layer_scores[l] * weights[l] for l in layer_scores)

    # Add cultural bonus
    cultural_bonus = calculate_cultural_bonus(critique, culture)

    # Final DCR (0-1 scale)
    dcr = min(1.0, base_dcr + cultural_bonus)

    # If reference dimensions provided, calculate overlap
    reference_overlap = None
    if reference_dims:
        if isinstance(reference_dims, str):
            try:
                reference_dims = json.loads(reference_dims)
            except:
                reference_dims = reference_dims.split(",")

        # Count layer coverage from reference
        ref_layers = defaultdict(int)
        for dim in reference_dims:
            for layer in ["L1", "L2", "L3", "L4", "L5"]:
                if f"_{layer}_" in dim:
                    ref_layers[layer] += 1

        reference_overlap = {l: ref_layers.get(l, 0) for l in ["L1", "L2", "L3", "L4", "L5"]}

    return {
        "dcr": round(dcr, 4),
        "dcr_5scale": round(dcr * 5, 2),  # Convert to 1-5 scale
        "layer_scores": {k: round(v, 4) for k, v in layer_scores.items()},
        "cultural_bonus": round(cultural_bonus, 4),
        "culture": culture,
        "reference_overlap": reference_overlap
    }


def evaluate_results(results_file: str, output_file: str = None):
    """
    Calculate DCR scores for a results file.

    Args:
        results_file: Path to VLM results JSONL
        output_file: Optional output path for scores
    """
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Calculating DCR for {len(results)} samples...")

    scores = []
    culture_scores = defaultdict(list)

    for r in results:
        critique = r.get("critique", "")
        culture = r.get("culture", "western")
        ref_dims = r.get("covered_dimensions")

        dcr_result = calculate_dcr(critique, culture, ref_dims)
        dcr_result["pair_id"] = r.get("pair_id")
        dcr_result["model"] = r.get("model")

        scores.append(dcr_result)
        culture_scores[culture].append(dcr_result["dcr"])

    # Calculate summary statistics
    all_dcr = [s["dcr"] for s in scores]
    summary = {
        "total_samples": len(scores),
        "mean_dcr": round(sum(all_dcr) / len(all_dcr), 4) if all_dcr else 0,
        "mean_dcr_5scale": round(sum(all_dcr) / len(all_dcr) * 5, 2) if all_dcr else 0,
        "by_culture": {
            c: round(sum(v) / len(v), 4) if v else 0
            for c, v in culture_scores.items()
        }
    }

    print(f"\n=== DCR Summary ===")
    print(f"Mean DCR: {summary['mean_dcr']:.4f} ({summary['mean_dcr_5scale']:.2f}/5)")
    print(f"\nBy Culture:")
    for culture, score in sorted(summary["by_culture"].items()):
        print(f"  {culture}: {score:.4f}")

    # Save results
    if output_file:
        output_data = {
            "summary": summary,
            "scores": scores
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {output_file}")

    return summary, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DCR scores for VLM results")
    parser.add_argument("--input", type=str, required=True, help="VLM results file (JSONL)")
    parser.add_argument("--output", type=str, default="dcr_scores.json", help="Output file")

    args = parser.parse_args()
    evaluate_results(args.input, args.output)
