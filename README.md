# 📚 Book Cover Recommendation System

A cross-modal retrieval system that takes a book cover image as input and returns ranked recommendations across four similarity dimensions: visual style, cross-modal (text→image), semantic theme, and color palette.

Built as a semester project for a **Natural Language Processing & Computer Vision** course.

---

## How It Works

A query cover image is processed through two parallel tracks — visual and textual — whose outputs are compared against a pre-built index of book covers using cosine similarity.

| Track | Signal | Method |
|---|---|---|
| 🖼️ Visual similarity | Cover art, composition, style | CLIP image → image |
| 🔀 Cross-modal | Text on cover → visual match | CLIP text (OCR) → image |
| 📖 Semantic / theme | Title, author, description | SentenceTransformer text → text |
| 🎨 Color palette | Dominant colors | k-means RGB → cosine similarity |

---

## Demo

```bash
# Query a single cover from the terminal
uv run python main.py data/covers_nyt/horror_9780385121675.jpg --embeddings embeddings_nyt

# Launch the web UI
uv run python ui.py
```

The web UI lets you select a cover from a dropdown, view the query image alongside OCR output, and browse results tab by tab. Clicking any result shows its full index metadata.

---

## Stack

| Component | Library / Model |
|---|---|
| Visual encoder | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Text encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| OCR | EasyOCR |
| Color extraction | scikit-learn k-means |
| Similarity search | NumPy cosine similarity |
| Demo UI | Gradio |
| Package manager | uv |

---

## Project Structure

```
book-rec/
├── data/
│   ├── covers_nyt/          # downloaded cover images
│   ├── nyt_index.json       # book metadata index
│   └── nyt_golden_set.json  # evaluation queries
├── embeddings_nyt/          # precomputed .npy vectors
│   ├── clip_image.npy       # (N, 512) CLIP image vectors
│   ├── clip_text.npy        # (N, 512) CLIP text vectors
│   ├── sentence.npy         # (N, 384) SentenceTransformer vectors
│   ├── colors.npy           # (N, 15)  color palette vectors
│   └── row_index.json       # row → book metadata mapping
├── src/
│   ├── data/                # dataset download scripts
│   ├── features/            # feature extractors + build_embeddings.py
│   ├── search/              # retrieval.py
│   └── eval/                # evaluate.py, visualize.py
├── main.py                  # terminal end-to-end test
├── ui.py                    # Gradio web interface
├── decisions.md             # documented technical decisions
└── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install uv
uv sync

# Build embeddings (run once, ~15 min)
uv run python src/features/build_embeddings.py \
    --index data/nyt_index.json \
    --output embeddings_nyt

# Evaluate against golden set
uv run python src/eval/evaluate.py \
    --golden data/nyt_golden_set.json \
    --embeddings embeddings_nyt
```

---

## Dataset

The primary dataset is sourced from:
- **NYT Bestsellers API** — current and recent bestseller lists across fiction and non-fiction
- **Google Books API** — genre-specific books anchored to well-known authors
- **Manual ISBNs** — curated classics for underrepresented genres (horror, biography)

~256 books across 10 genres. Covers downloaded as JPEG images. Metadata includes title, author, genre, publisher, and NYT rank where available.

---

## Evaluation

Precision@3 (P@3) on a manually curated golden set of 10 queries.

**Baseline results (ViT-B/32 + EasyOCR + MiniLM-L6-v2):**

| Track | Mean P@3 |
|---|---|
| Visual | 0.47 |
| Semantic | 0.43 |
| Cross-modal | 0.33 |
| Color | 0.10 |

---

## Experiments (In Progress)

| | Visual P@3 | Cross-modal P@3 | Semantic P@3 | Color P@3 |
|---|---|---|---|---|
| ViT-B/32 + OCR (baseline) | 0.47 | 0.33 | 0.43 | 0.10 |
| ViT-B/16 + OCR | — | — | — | — |
| ViT-B/32 + Qwen2-VL | — | — | — | — |
| ViT-B/16 + Qwen2-VL | — | — | — | — |

---

## Related Work

This project extends the findings of:

> Toosi et al., *"Unlocking Book Genre from Covers: A Multimodal Approach to Book Genre Prediction"*, IJWR 2025.

That paper demonstrated that replacing OCR text with VLM-generated descriptions improves book cover **classification** accuracy from 52.98% to 63.31% (Top-1). We investigate whether the same holds for cross-modal **retrieval**.