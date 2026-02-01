"""
VULCA-Bench Quick Start Example
Demonstrates basic usage of the benchmark.
"""

import json
import sys
sys.path.append('..')

from evaluation.calculate_dcr import calculate_dcr
from evaluation.layer_scorer import score_all_layers


def load_samples(filepath: str, n: int = 5):
    """Load first n samples from dataset."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))
    return samples


def main():
    print("=== VULCA-Bench Quick Start ===\n")

    # Load sample data
    samples = load_samples("../data/vulca_bench.jsonl", n=3)

    for i, sample in enumerate(samples):
        print(f"--- Sample {i+1}: {sample['pair_id']} ---")
        print(f"Culture: {sample['culture']}")
        print(f"Artist: {sample.get('artist', 'Unknown')}")
        print(f"Title: {sample.get('title', 'Untitled')}")

        # Calculate DCR on reference critique
        critique_en = sample.get('critique_en', '')
        if critique_en:
            dcr_result = calculate_dcr(
                critique=critique_en,
                culture=sample['culture'],
                reference_dims=sample.get('covered_dimensions')
            )
            print(f"\nDCR Score: {dcr_result['dcr']:.4f} ({dcr_result['dcr_5scale']:.2f}/5)")
            print(f"Layer Coverage: {dcr_result['layer_scores']}")

            # Layer scores
            layer_result = score_all_layers(critique_en)
            print(f"Layer Scores: L1={layer_result['layers']['L1']['score']:.1f}, "
                  f"L2={layer_result['layers']['L2']['score']:.1f}, "
                  f"L3={layer_result['layers']['L3']['score']:.1f}, "
                  f"L4={layer_result['layers']['L4']['score']:.1f}, "
                  f"L5={layer_result['layers']['L5']['score']:.1f}")
            print(f"Overall: {layer_result['overall_score']:.2f}/5")

        print()


if __name__ == "__main__":
    main()
