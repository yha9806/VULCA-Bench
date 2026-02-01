"""
VULCA-Bench Layer Scorer
Scores VLM critiques on each of the 5 cultural understanding layers (L1-L5).

Layers:
- L1: Visual Perception (colors, lines, composition)
- L2: Technical Analysis (medium, technique, materials)
- L3: Cultural Symbolism (motifs, iconography)
- L4: Historical Context (period, artist, provenance)
- L5: Philosophical Aesthetics (theory, values)

Usage:
    python layer_scorer.py --input results/gpt-4o_results.jsonl --output layer_scores.json

Author: VULCA Project Team
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


# =============================================================================
# Layer Definitions
# =============================================================================

LAYER_DEFINITIONS = {
    "L1": {
        "name": "Visual Perception",
        "description": "Basic visual elements: color, line, form, composition",
        "keywords": [
            # English
            "color", "colour", "hue", "tone", "shade", "tint",
            "line", "curve", "contour", "outline", "edge",
            "composition", "arrangement", "layout", "balance", "symmetry",
            "shape", "form", "silhouette", "figure", "ground",
            "light", "shadow", "contrast", "brightness", "darkness",
            "texture", "pattern", "rhythm", "movement", "flow",
            # Chinese
            "颜色", "色彩", "色调", "线条", "轮廓", "构图",
            "形状", "形态", "光影", "明暗", "对比", "纹理", "节奏"
        ],
        "weight": 0.20
    },
    "L2": {
        "name": "Technical Analysis",
        "description": "Medium, technique, materials, craftsmanship",
        "keywords": [
            # English
            "brushwork", "brush stroke", "technique", "method",
            "oil", "watercolor", "ink", "tempera", "acrylic", "gouache",
            "canvas", "silk", "paper", "wood", "panel",
            "pigment", "paint", "medium", "material",
            "layer", "glaze", "wash", "impasto", "sfumato",
            "stroke", "application", "execution", "rendering",
            # Chinese
            "笔法", "笔触", "技法", "技巧", "媒材", "颜料",
            "油彩", "水墨", "绢本", "纸本", "设色", "渲染", "皴法"
        ],
        "weight": 0.20
    },
    "L3": {
        "name": "Cultural Symbolism",
        "description": "Motifs, iconography, symbolic meanings",
        "keywords": [
            # English
            "symbol", "symbolism", "symbolic", "motif",
            "iconography", "icon", "emblem", "attribute",
            "allegory", "allegorical", "metaphor", "represent",
            "meaning", "significance", "tradition", "traditional",
            "mythology", "mythological", "legend", "narrative",
            "religious", "sacred", "spiritual", "ritual",
            "auspicious", "fortune", "blessing",
            # Chinese
            "象征", "图像", "寓意", "典故", "传统", "文化",
            "神话", "宗教", "仪式", "吉祥", "祥瑞", "隐喻"
        ],
        "weight": 0.20
    },
    "L4": {
        "name": "Historical Context",
        "description": "Period, artist, provenance, art movements",
        "keywords": [
            # English
            "dynasty", "period", "era", "century", "date", "year",
            "artist", "painter", "master", "school", "workshop",
            "movement", "style", "influence", "influenced",
            "patron", "commission", "collection", "provenance",
            "renaissance", "baroque", "romantic", "impressionist",
            "ming", "qing", "song", "tang", "yuan",
            # Chinese
            "朝代", "时期", "年代", "世纪", "画家", "大师",
            "流派", "风格", "影响", "赞助", "收藏", "传承"
        ],
        "weight": 0.20
    },
    "L5": {
        "name": "Philosophical Aesthetics",
        "description": "Aesthetic theory, cultural values, philosophical concepts",
        "keywords": [
            # English
            "philosophy", "philosophical", "aesthetic", "aesthetics",
            "beauty", "beautiful", "sublime", "transcendent",
            "spiritual", "spirit", "essence", "soul",
            "harmony", "harmonious", "unity", "wholeness",
            "meaning", "interpretation", "significance",
            "emotion", "feeling", "expression", "evoke",
            "theory", "principle", "concept", "idea",
            # Chinese - Core aesthetic concepts
            "气韵", "意境", "神韵", "境界", "精神", "灵性",
            "哲学", "美学", "和谐", "统一", "超越", "感悟",
            # Japanese aesthetic concepts
            "wabi", "sabi", "yugen", "aware",
            # Indian aesthetic concepts
            "rasa", "bhava"
        ],
        "weight": 0.20
    }
}


# =============================================================================
# Scoring Functions
# =============================================================================

def count_keyword_matches(text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    """Count keyword matches and return matched keywords."""
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        # Handle multi-word keywords and Chinese
        if kw.lower() in text_lower:
            matched.append(kw)
    return len(matched), matched


def score_layer(critique: str, layer: str) -> Dict[str, Any]:
    """
    Score a single layer for a critique.

    Returns:
        Dict with score (1-5), matches, and evidence
    """
    if not critique:
        return {"score": 1.0, "matches": 0, "evidence": []}

    layer_def = LAYER_DEFINITIONS.get(layer)
    if not layer_def:
        return {"score": 1.0, "matches": 0, "evidence": []}

    keywords = layer_def["keywords"]
    num_matches, matched_keywords = count_keyword_matches(critique, keywords)

    # Scoring rubric (1-5 scale)
    # 1: No coverage (0 matches)
    # 2: Minimal (1-2 matches)
    # 3: Moderate (3-5 matches)
    # 4: Good (6-10 matches)
    # 5: Excellent (11+ matches)

    if num_matches == 0:
        score = 1.0
    elif num_matches <= 2:
        score = 2.0
    elif num_matches <= 5:
        score = 3.0
    elif num_matches <= 10:
        score = 4.0
    else:
        score = 5.0

    return {
        "score": score,
        "matches": num_matches,
        "evidence": matched_keywords[:10]  # Top 10 matches
    }


def score_all_layers(critique: str) -> Dict[str, Any]:
    """
    Score all 5 layers for a critique.

    Returns:
        Dict with per-layer scores and overall score
    """
    layer_results = {}
    for layer in ["L1", "L2", "L3", "L4", "L5"]:
        layer_results[layer] = score_layer(critique, layer)

    # Calculate weighted average
    weights = {l: LAYER_DEFINITIONS[l]["weight"] for l in LAYER_DEFINITIONS}
    weighted_sum = sum(layer_results[l]["score"] * weights[l] for l in layer_results)

    # Check for inverted pyramid (L5 > L1-L4 average) - potential pseudo-understanding
    l1_l4_avg = sum(layer_results[l]["score"] for l in ["L1", "L2", "L3", "L4"]) / 4
    l5_score = layer_results["L5"]["score"]
    inverted = l5_score > l1_l4_avg + 0.5  # L5 significantly higher than foundation

    return {
        "layers": layer_results,
        "overall_score": round(weighted_sum, 2),
        "pyramid_inverted": inverted,
        "l1_l4_avg": round(l1_l4_avg, 2),
        "l5_score": l5_score
    }


def evaluate_file(input_file: str, output_file: str = None):
    """
    Evaluate all samples in a results file.
    """
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Scoring {len(results)} samples across L1-L5 layers...")

    scored_results = []
    culture_stats = defaultdict(lambda: defaultdict(list))

    for r in results:
        critique = r.get("critique", "")
        culture = r.get("culture", "unknown")

        layer_scores = score_all_layers(critique)
        layer_scores["pair_id"] = r.get("pair_id")
        layer_scores["model"] = r.get("model")
        layer_scores["culture"] = culture

        scored_results.append(layer_scores)

        # Aggregate by culture
        for layer in ["L1", "L2", "L3", "L4", "L5"]:
            culture_stats[culture][layer].append(layer_scores["layers"][layer]["score"])
        culture_stats[culture]["overall"].append(layer_scores["overall_score"])

    # Summary statistics
    summary = {
        "total_samples": len(scored_results),
        "inverted_pyramid_count": sum(1 for r in scored_results if r["pyramid_inverted"]),
        "by_culture": {}
    }

    for culture, layers in culture_stats.items():
        summary["by_culture"][culture] = {
            layer: round(sum(scores) / len(scores), 2) if scores else 0
            for layer, scores in layers.items()
        }

    # Print summary
    print(f"\n=== Layer Score Summary ===")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Inverted Pyramid (potential pseudo-understanding): {summary['inverted_pyramid_count']}")

    print(f"\nScores by Culture (L1-L5 + Overall):")
    for culture, scores in sorted(summary["by_culture"].items()):
        layers_str = " ".join(f"L{i}:{scores.get(f'L{i}', 0):.1f}" for i in range(1, 6))
        print(f"  {culture:12s}: {layers_str} | Overall: {scores.get('overall', 0):.2f}")

    # Save results
    if output_file:
        output_data = {
            "summary": summary,
            "scores": scored_results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {output_file}")

    return summary, scored_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score VLM results on L1-L5 layers")
    parser.add_argument("--input", type=str, required=True, help="VLM results file (JSONL)")
    parser.add_argument("--output", type=str, default="layer_scores.json", help="Output file")

    args = parser.parse_args()
    evaluate_file(args.input, args.output)
