# 📚 Book Cover Recommendation System

A cross-modal retrieval system that takes a book cover image as input and returns ranked recommendations across four similarity dimensions: visual style, cross-modal (text→image), semantic theme, and color palette.

Built as a semester project for a **Natural Language Processing & Computer Vision** course. Compares 7 model configurations (CLIP, OpenCLIP, SigLIP 2, with OCR and VLM-based text enrichment) using zero-shot embedding-based retrieval — no fine-tuning on book covers.

---

## How It Works

A query cover image is processed through two parallel tracks — visual and textual — whose outputs are compared against a pre-built index of book covers using cosine similarity.

| Track | Signal | Method |
|---|---|---|
| 🖼️ Visual similarity | Cover art, composition, style | CLIP/SigLIP image → image |
| 🔀 Cross-modal | Text on cover → visual match | CLIP/SigLIP text (OCR or VLM) → image |
| 📖 Semantic / theme | Title, author, description | SentenceTransformer text → text |
| 🎨 Color palette | Dominant colors | k-means LAB → cosine similarity |

All models are used **zero-shot** — pretrained encoders applied directly via cosine similarity search, with no task-specific fine-tuning.

---

## Demo

```bash
# Query a single cover from the terminal
uv run python main.py data/covers_nyt/horror_9780385121675.jpg --embeddings embeddings_nyt

# Launch the web UI
uv run python ui.py
```

The web UI supports:
- **Model selector** — switch live between 7 configurations (CLIP B/32, B/16, OpenCLIP, SigLIP 2, each with OCR or VLM text)
- **Dataset browsing** — select any of the 173 indexed covers
- **External upload** — drop in any book cover image and get recommendations against the indexed dataset
- **Smart text source** — automatically uses OCR or live VLM description depending on the active model, matching how the index was built

---

## Stack

| Component | Library / Model |
|---|---|
| Visual encoders | CLIP (ViT-B/32, ViT-B/16), OpenCLIP (LAION-2B), SigLIP 2 |
| Text encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| OCR | EasyOCR |
| VLM descriptions | Qwen2-VL 2B Instruct |
| Color extraction | scikit-learn k-means + scikit-image (LAB) |
| Similarity search | NumPy cosine similarity |
| Demo UI | Gradio |
| Package manager | uv |

---

## Project Structure

```
book-rec/
├── data/
│   ├── covers_nyt/          # 173 downloaded cover images
│   ├── nyt_index.json       # book metadata index (incl. OCR text / VLM descriptions)
│   └── nyt_golden_set.json  # 7 evaluation queries
├── embeddings_nyt*/         # precomputed .npy vectors, one dir per experiment config
│   ├── clip_image.npy
│   ├── clip_text.npy
│   ├── sentence.npy
│   ├── colors.npy
│   └── row_index.json
├── src/
│   ├── data/                # dataset download + cleanup scripts
│   ├── features/            # feature extractors + build_embeddings.py
│   ├── search/               # retrieval.py
│   └── eval/                 # evaluate.py, visualize.py
├── main.py                  # terminal end-to-end test
├── ui.py                    # Gradio web interface
├── decisions.md             # full technical decisions + experiment results
├── presentation.pptx        # final presentation deck
└── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install uv
uv sync

# Build embeddings for a configuration (run once per config)
uv run python src/features/build_embeddings.py \
    --index data/nyt_index.json \
    --output embeddings_nyt \
    --clip-model openai/clip-vit-base-patch32

# Evaluate against golden set
uv run python src/eval/evaluate.py \
    --golden data/nyt_golden_set.json \
    --embeddings embeddings_nyt
```

**Supported `--clip-model` values:**
```
openai/clip-vit-base-patch32
openai/clip-vit-base-patch16
openclip:ViT-B-32/laion2b_s34b_b79k
siglip:google/siglip2-base-patch16-224
```

**Text source** (`--text-source`, default `ocr`):
```bash
# Use VLM descriptions instead of OCR (requires running vlm_describe.py first)
uv run python src/features/vlm_describe.py --index data/nyt_index.json
uv run python src/features/build_embeddings.py --index data/nyt_index.json --output embeddings_nyt_vlm --text-source vlm
```

---

## Dataset

The dataset is sourced from:
- **NYT Bestsellers API** — current and recent bestseller lists across fiction and non-fiction
- **Google Books API** — genre-specific books anchored to well-known authors
- **Manual ISBNs** — curated classics for underrepresented genres (horror, biography)

**173 books** across 10 genres after substantial data quality cleanup — removing 106 placeholder/low-quality covers (Google Books "image not available" thumbnails, audiobook edition duplicates, scanned-text-only covers identified through manual review).

---

## Evaluation

Precision@3 (P@3) on a manually curated golden set of 7 queries, built via relevance judging (expected matches derived from actual retrieval results, not genre labels alone).

**Final results — clean 173-book dataset:**

| Configuration | Visual | Cross-modal | Semantic | Color |
|---|---|---|---|---|
| **Baseline:** ViT-B/32 + OCR | 0.48 | 0.48 | **0.52** | 0.05 |
| ViT-B/16 + OCR | 0.48 | 0.48 | 0.52 | 0.05 |
| ViT-B/32 + Qwen2-VL | 0.48 | 0.48 | 0.43 ↓ | 0.05 |
| ViT-B/16 + Qwen2-VL | 0.48 | 0.48 | 0.43 ↓ | 0.05 |
| OpenCLIP + OCR | 0.43 | **0.52** | 0.52 | 0.05 |
| OpenCLIP + Qwen2-VL | 0.43 | **0.52** | 0.43 ↓ | 0.05 |
| SigLIP 2 + OCR | 0.38 ↓ | 0.00 ↓↓ | 0.52 | 0.05 |

See `decisions.md` for full experiment writeups, mechanistic explanations, and discussion of result stability across dataset versions.

---

## Key Findings

1. **Data quality dominates model choice** — removing 106 junk covers improved retrieval more than any model swap
2. **VLM descriptions hurt retrieval** — contrary to classification literature, sparse precise OCR text outperforms rich generic VLM descriptions for cosine-similarity search
3. **No single model wins all tracks** — the optimal configuration depends on which track matters most
4. **Training objective shapes the embedding space** — SigLIP 2's sigmoid loss makes cross-modal retrieval fail entirely (P@3 = 0.00), despite strong visual and semantic performance
5. **Color palette is a fundamental dead end** — two extraction methods (RGB, LAB) both plateau at the same low ceiling; color similarity does not predict book similarity

---

## Related Work

This project extends the findings of:

> Toosi et al., *"Unlocking Book Genre from Covers: A Multimodal Approach to Book Genre Prediction"*, IJWR 2025.

That paper demonstrated that replacing OCR text with VLM-generated descriptions improves book cover **classification** accuracy from 52.98% to 63.31% (Top-1). We investigate whether the same holds for cross-modal **retrieval** — and find the opposite: VLM enrichment consistently *hurts* retrieval performance, highlighting a fundamental difference between classification (which benefits from rich discriminative features) and retrieval (which benefits from sparse, precise signals).
