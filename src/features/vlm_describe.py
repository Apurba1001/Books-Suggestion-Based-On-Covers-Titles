"""
vlm_describe.py
---------------
Generates rich text descriptions of book covers using Qwen2-VL 2B.
Saves descriptions back into the index JSON file.

This is a one-time offline preprocessing step — run once, then
build_embeddings.py can use the descriptions instead of OCR text.

Usage:
    uv run python src/features/vlm_describe.py --index data/nyt_index.json

Output:
    Adds a "vlm_description" field to each book in the index JSON.
    Books that already have a vlm_description are skipped (safe to re-run).
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT = (
    "This is a book cover. Describe it in a paragraph. "
    "Include: the visual style and mood of the cover, the dominant colors, "
    "any visible text (title, author, taglines), the type of imagery or "
    "illustrations used, and what genre or audience the cover suggests."
)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/nyt_index.json",
                        help="Path to index JSON file")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate descriptions even if they exist")
    args = parser.parse_args()

    index_path = Path(args.index)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    # Count how many need descriptions
    needs_desc = [
        b for b in index
        if args.force or not b.get("vlm_description")
    ]
    print(f"Total books   : {len(index)}")
    print(f"Need VLM desc : {len(needs_desc)}")

    if not needs_desc:
        print("All books already have descriptions. Use --force to regenerate.")
        return

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.\n")

    # Generate descriptions
    for i, book in enumerate(index):
        if not args.force and book.get("vlm_description"):
            continue

        img_path = book["filename"]
        if not Path(img_path).exists():
            print(f"[{i+1:03d}] SKIP (file missing) {book['title'][:50]}")
            book["vlm_description"] = ""
            continue

        print(f"[{i+1:03d}/{len(index)}] {book['title'][:50]:<50} ", end="", flush=True)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{Path(img_path).resolve()}"},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                )

            # Decode only the generated tokens (skip the prompt)
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            description = processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()

            book["vlm_description"] = description
            print(f"✓ ({len(description)} chars)")

        except Exception as e:
            print(f"✗ Error: {e}")
            book["vlm_description"] = ""

    # Save enriched index
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    described = sum(1 for b in index if b.get("vlm_description"))
    print(f"\n✓ Done. {described}/{len(index)} books now have VLM descriptions.")
    print(f"  Saved to: {index_path}")


if __name__ == "__main__":
    main()