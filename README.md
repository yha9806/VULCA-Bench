# VULCA-Bench

> [!IMPORTANT]
> **This repository is a historical snapshot.** Active development and the
> canonical public release have moved to
> [vulca-org/vulca-cultural-visual-benchmark](https://github.com/vulca-org/vulca-cultural-visual-benchmark).
> Use the organization repository for current data, validation, issues, and
> contributions.
>
> The canonical metadata-first v2.1 release contains 7,236 records and
> redistributes no artwork image files. The files retained here contain 7,410
> paper-era records, while the legacy documentation below reports 7,408; neither
> count describes v2.1. See the canonical
> [release reconciliation](https://github.com/vulca-org/vulca-cultural-visual-benchmark/blob/main/RELEASES.md)
> and [image-rights boundary](https://github.com/vulca-org/vulca-cultural-visual-benchmark/blob/main/IMAGE_RIGHTS.md)
> before using the benchmark.

**VULCA-Bench** is a comprehensive benchmark for evaluating Vision-Language Models (VLMs) on multi-cultural art critique tasks. It provides 7,408 artwork samples across 8 cultural traditions with expert-annotated critiques and dimensional coverage.

📄 **Paper**: [VULCA-Bench: A Multi-Cultural Art Critique Benchmark for VLMs](https://arxiv.org/abs/2601.07986)

🤗 **Dataset**: [HuggingFace Datasets](https://huggingface.co/datasets/harryHURRY/vulca-bench)

## Dataset Overview

| Culture | Samples | Description |
|---------|---------|-------------|
| Western | 4,041 | European/American art (Renaissance to Modern) |
| Chinese | 2,041 | Traditional Chinese painting (ink wash, gongbi, etc.) |
| Japanese | 401 | Ukiyo-e, Nihonga, Rinpa traditions |
| Islamic | 205 | Persian miniatures, geometric patterns, calligraphy |
| Mural | 200 | Cave paintings, frescoes (Dunhuang, Ajanta, etc.) |
| Hermitage | 196 | European masterworks from the Hermitage Museum |
| Indian | 173 | Mughal, Rajput, Pahari miniatures |
| Korean | 151 | Minhwa, literati painting, Joseon court art |
| **Total** | **7,408** | |

## 5-Layer Cultural Understanding Framework (L1-L5)

Each artwork is annotated with dimensions across 5 cultural understanding layers:

| Layer | Name | Description |
|-------|------|-------------|
| **L1** | Visual Perception | Color, line, composition, visual elements |
| **L2** | Technical Analysis | Medium, technique, materials, craftsmanship |
| **L3** | Cultural Symbolism | Motifs, iconography, symbolic meanings |
| **L4** | Historical Context | Period, artist, provenance, art movements |
| **L5** | Philosophical Aesthetics | Aesthetic theory, cultural values, philosophy |

## Quick Start

### 1. Install Dependencies

```bash
pip install openai anthropic google-generativeai pillow
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

### 3. Run VLM Evaluation

```bash
# Generate critiques with GPT-4o
python evaluation/run_vlm.py --model gpt-4o --input data/vulca_bench.jsonl --output results/

# Calculate DCR (Dimension Coverage Rate)
python evaluation/calculate_dcr.py --input results/gpt-4o_results.jsonl --output dcr_scores.json

# Score across L1-L5 layers
python evaluation/layer_scorer.py --input results/gpt-4o_results.jsonl --output layer_scores.json
```

## Data Format

Each sample in `vulca_bench.jsonl`:

```json
{
  "pair_id": "PAIR_00001",
  "ulid": "01JKAB1234567890ABCDEF",
  "culture": "chinese",
  "image_path": "/path/to/artwork.jpg",
  "artist": "Qi Baishi",
  "title": "Shrimp",
  "critique_zh": "齐白石的《虾》...",
  "critique_en": "Qi Baishi's 'Shrimp'...",
  "covered_dimensions": ["CN_L1_D1", "CN_L2_D3", ...],
  "art_form": "painting",
  "art_style": "ink_wash",
  "art_genre": "animal"
}
```

## Evaluation Metrics

### DCR (Dimension Coverage Rate)
Measures how many expert-annotated dimensions are covered by the VLM's critique.

```python
from evaluation.calculate_dcr import calculate_dcr

result = calculate_dcr(
    critique="The artwork displays masterful brushwork...",
    culture="chinese",
    reference_dims=["CN_L1_D1", "CN_L2_D3"]
)
print(f"DCR: {result['dcr']:.4f}")  # 0-1 scale
```

### Layer Scores (L1-L5)
Individual scores for each cultural understanding layer.

```python
from evaluation.layer_scorer import score_all_layers

scores = score_all_layers(critique)
print(f"L1: {scores['layers']['L1']['score']}")  # 1-5 scale
print(f"Overall: {scores['overall_score']}")
```

## Culture-Specific Dimensions

| Culture | Prefix | Total Dims | 70% Threshold |
|---------|--------|------------|---------------|
| Chinese | CN_ | 30 | ≥21 |
| Western | WE_ | 27 | ≥19 |
| Japanese | JP_ | 28 | ≥20 |
| Korean | KR_ | 26 | ≥18 |
| Islamic | IS_ | 32 | ≥22 |
| Indian | IN_ | 33 | ≥23 |
| Hermitage | WS_ | 30 | ≥21 |
| Mural | MU_ | 30 | ≥21 |

## Repository Structure

```
VULCA-Bench/
├── data/
│   ├── vulca_bench.jsonl          # Full dataset (7,408 samples)
│   └── culture_subsets/           # Per-culture splits
│       ├── chinese.jsonl
│       ├── western.jsonl
│       ├── japanese.jsonl
│       └── ...
├── evaluation/
│   ├── run_vlm.py                 # VLM evaluation runner
│   ├── calculate_dcr.py           # DCR calculation
│   └── layer_scorer.py            # L1-L5 layer scoring
├── examples/
│   └── quick_start.py             # Usage examples
└── README.md
```

## Citation

```bibtex
@article{yu2026vulcabench,
  title={VULCA-Bench: A Multi-Cultural Art Critique Benchmark for Vision-Language Models},
  author={Yu, Haorui and Yang, Diji and He, Hang and Zhang, Fengrui and Yi, Qiufeng},
  journal={arXiv preprint arXiv:2601.07986},
  year={2026}
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

We thank the Hermitage Museum, National Palace Museum, and various cultural institutions for providing artwork images and metadata.
