"""
VULCA-Bench VLM Evaluation Runner
Generates art critiques using various Vision-Language Models.

Usage:
    python run_vlm.py --model gpt-4o --input data/vulca_bench.jsonl --output results/

Supported Models:
    - gpt-4o, gpt-4o-mini (OpenAI)
    - claude-3.5-sonnet, claude-3-opus (Anthropic)
    - gemini-2.0-flash, gemini-1.5-pro (Google)

Author: VULCA Project Team
"""

import argparse
import json
import os
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional

# API Clients (install with: pip install openai anthropic google-generativeai)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# =============================================================================
# Prompt Template
# =============================================================================

CRITIQUE_PROMPT = """As an art historian, provide a comprehensive critique of this artwork.

Analyze across five dimensions:
1. **Visual Perception (L1)**: Describe colors, lines, composition, and visual elements.
2. **Technical Analysis (L2)**: Identify the medium, techniques, and materials used.
3. **Cultural Symbolism (L3)**: Explain cultural motifs, iconography, and symbolic meanings.
4. **Historical Context (L4)**: Place the work in its historical period and artistic tradition.
5. **Philosophical Aesthetics (L5)**: Discuss the deeper philosophical or aesthetic values.

Provide the critique in English, using appropriate art historical terminology.
"""


# =============================================================================
# VLM Clients
# =============================================================================

def encode_image_base64(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_openai(image_path: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Generate critique using OpenAI GPT-4V."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = openai.OpenAI()
    base64_image = encode_image_base64(image_path)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CRITIQUE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=2000,
        temperature=0.7
    )
    latency = time.time() - start_time

    return {
        "critique": response.choices[0].message.content,
        "model": model,
        "tokens": response.usage.total_tokens,
        "latency": latency
    }


def generate_anthropic(image_path: str, model: str = "claude-3-5-sonnet-20241022") -> Dict[str, Any]:
    """Generate critique using Anthropic Claude."""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic()
    base64_image = encode_image_base64(image_path)

    # Detect media type
    suffix = Path(image_path).suffix.lower()
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix[1:], "image/jpeg")

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": base64_image}
                    },
                    {"type": "text", "text": CRITIQUE_PROMPT}
                ]
            }
        ]
    )
    latency = time.time() - start_time

    return {
        "critique": response.content[0].text,
        "model": model,
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
        "latency": latency
    }


def generate_google(image_path: str, model: str = "gemini-2.0-flash-exp") -> Dict[str, Any]:
    """Generate critique using Google Gemini."""
    if not GOOGLE_AVAILABLE:
        raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model_client = genai.GenerativeModel(model)

    from PIL import Image
    img = Image.open(image_path)

    start_time = time.time()
    response = model_client.generate_content([CRITIQUE_PROMPT, img])
    latency = time.time() - start_time

    return {
        "critique": response.text,
        "model": model,
        "tokens": 0,  # Gemini doesn't expose token count directly
        "latency": latency
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_evaluation(
    input_file: str,
    output_dir: str,
    model: str = "gpt-4o",
    limit: Optional[int] = None
):
    """
    Run VLM evaluation on VULCA-Bench dataset.

    Args:
        input_file: Path to vulca_bench.jsonl
        output_dir: Directory to save results
        model: Model identifier
        limit: Optional limit on number of samples
    """
    # Select generator function
    if model.startswith("gpt"):
        generate_fn = lambda p: generate_openai(p, model)
    elif model.startswith("claude"):
        generate_fn = lambda p: generate_anthropic(p, model)
    elif model.startswith("gemini"):
        generate_fn = lambda p: generate_google(p, model)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Load dataset
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    if limit:
        samples = samples[:limit]

    print(f"Evaluating {len(samples)} samples with {model}...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{model}_results.jsonl"

    results = []
    for i, sample in enumerate(samples):
        pair_id = sample.get("pair_id", f"sample_{i}")
        image_path = sample.get("optimized_path") or sample.get("image_path")

        if not image_path or not Path(image_path).exists():
            print(f"  Skipping {pair_id}: image not found")
            continue

        try:
            result = generate_fn(image_path)
            result["pair_id"] = pair_id
            result["culture"] = sample.get("culture")
            result["reference_zh"] = sample.get("critique_zh")
            result["reference_en"] = sample.get("critique_en")
            result["covered_dimensions"] = sample.get("covered_dimensions")

            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)}")

        except Exception as e:
            print(f"  Error on {pair_id}: {e}")

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(results)} results to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM evaluation on VULCA-Bench")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--input", type=str, default="data/vulca_bench.jsonl", help="Input dataset")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")

    args = parser.parse_args()
    run_evaluation(args.input, args.output, args.model, args.limit)
