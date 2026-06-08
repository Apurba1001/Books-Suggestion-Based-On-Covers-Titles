"""
ui.py
-----
Simple UI to view recommendation results with cover images.
Select a dataset, pick a cover, and see results with full index metadata on click.

Usage:
    uv run python main/ui.py
    uv run python main/ui.py --embeddings embeddings_nyt
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from src.features.clip_encoder import encode_image, encode_text
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder  import encode as encode_sentence
from src.search.retrieval       import load_index, search

TOP_K = 3

# ── Config ─────────────────────────────────────────────────────────────────────

DATASETS = {
    "OpenLibrary (~387 books)": ("embeddings",     "data/covers",     "data/index_updated.json",     "data/index.json"),
    "NYT Bestsellers (~260 books)": ("embeddings_nyt", "data/covers_nyt", "data/nyt_index_updated.json", "data/nyt_index.json"),
}


def load_full_index(json_path: str) -> dict:
    path = Path(json_path)
    if not path.exists():
        path = Path(json_path.replace("_updated", ""))
    with open(path, encoding="utf-8") as f:
        books = json.load(f)
    # key by isbn or cover_id
    return {str(b.get("isbn") or b.get("cover_id", "")): b for b in books}


def list_covers(covers_dir: str) -> list[str]:
    d = Path(covers_dir)
    if d.exists():
        return sorted([f.name for f in d.glob("*.jpg")])
    return []


# ── Core pipeline ──────────────────────────────────────────────────────────────

current_dataset   = None
current_full_index = {}
click_map_store   = {}


def switch_dataset(dataset_label: str):
    global current_dataset, current_full_index
    emb_dir, covers_dir, updated_idx, fallback_idx = DATASETS[dataset_label]
    load_index(emb_dir)
    current_full_index = load_full_index(updated_idx) or load_full_index(fallback_idx)
    current_dataset    = dataset_label
    choices = list_covers(covers_dir)
    return gr.update(choices=choices, value=None), f"Loaded **{dataset_label}**"


def recommend(dataset_label: str, filename: str):
    if not filename or not dataset_label:
        return None, "", [], [], [], [], {}, ""

    _, covers_dir, _, _ = DATASETS[dataset_label]
    path = Path(covers_dir) / filename
    if not path.exists():
        return None, f"File not found: {path}", [], [], [], [], {}, ""

    ocr_text     = extract_text(str(path))
    text         = ocr_text.strip() if ocr_text.strip() else path.stem.replace("_", " ")
    clip_img_vec = encode_image(str(path))
    clip_txt_vec = encode_text(text)
    sentence_vec = encode_sentence(text)
    color_vec    = extract_palette(str(path))

    results = search(
        clip_image_vec = clip_img_vec,
        clip_text_vec  = clip_txt_vec,
        sentence_vec   = sentence_vec,
        color_vec      = color_vec,
        k              = TOP_K,
    )

    ocr_info = f"**OCR:** {ocr_text}" if ocr_text.strip() else "**OCR:** nothing detected"

    click_map = {}

    def build_gallery(track):
        hits = results.get(track, [])
        gallery = []
        for i, hit in enumerate(hits):
            authors = ", ".join(hit["authors"]) if hit["authors"] else "Unknown"
            caption = f"#{hit['rank']}  {hit['title'][:40]}\nScore: {hit['score']:.3f} | {hit['genre']}\nby {authors}"
            gallery.append((hit["filename"], caption))
            click_map[f"{track}:{i}"] = str(hit["id"])
        return gallery

    return (
        str(path),
        ocr_info,
        build_gallery("visual"),
        build_gallery("cross_modal"),
        build_gallery("semantic"),
        build_gallery("color"),
        click_map,
        "",
    )


def on_click(track: str, click_map: dict, evt: gr.SelectData):
    key      = f"{track}:{evt.index}"
    book_id  = click_map.get(key)
    if not book_id:
        return "No metadata found."
    book = current_full_index.get(book_id)
    if not book:
        return f"ID `{book_id}` not found in index."
    lines = [f"### 📋 {book.get('title', '?')}"]
    lines.append("```json")
    lines.append(json.dumps(book, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Book Cover Recommender", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📚 Book Cover Recommender")
    gr.Markdown("*Select a dataset, pick a cover, and click any result to see its metadata.*")

    click_map_state = gr.State({})

    with gr.Row():
        dataset_dd = gr.Dropdown(
            choices=list(DATASETS.keys()),
            value=list(DATASETS.keys())[1],   # default to NYT
            label="Dataset",
        )
        dataset_status = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):
            cover_dd  = gr.Dropdown(choices=[], label="Select a cover", allow_custom_value=True)
            run_btn   = gr.Button("Find similar books", variant="primary")
            query_img = gr.Image(label="Query cover", height=300)
            ocr_out   = gr.Markdown("")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🖼️ Visual"):
                    g_visual = gr.Gallery(label="Similar cover style", columns=3, height=350, object_fit="contain")
                with gr.Tab("🔀 Cross-modal"):
                    g_cross = gr.Gallery(label="OCR text → image", columns=3, height=350, object_fit="contain")
                with gr.Tab("📖 Semantic"):
                    g_semantic = gr.Gallery(label="Similar theme", columns=3, height=350, object_fit="contain")
                with gr.Tab("🎨 Color"):
                    g_color = gr.Gallery(label="Similar palette", columns=3, height=350, object_fit="contain")

    detail_out = gr.Markdown("*Click a result to see its index entry.*")

    # Load default dataset on startup
    demo.load(fn=switch_dataset, inputs=[dataset_dd], outputs=[cover_dd, dataset_status])

    dataset_dd.change(fn=switch_dataset, inputs=[dataset_dd], outputs=[cover_dd, dataset_status])

    run_btn.click(
        fn=recommend,
        inputs=[dataset_dd, cover_dd],
        outputs=[query_img, ocr_out, g_visual, g_cross, g_semantic, g_color, click_map_state, detail_out],
    )

    def click_visual(state, evt: gr.SelectData):   return on_click("visual",      state, evt)
    def click_cross(state, evt: gr.SelectData):    return on_click("cross_modal", state, evt)
    def click_semantic(state, evt: gr.SelectData): return on_click("semantic",    state, evt)
    def click_color(state, evt: gr.SelectData):    return on_click("color",       state, evt)

    g_visual.select(click_visual,   inputs=[click_map_state], outputs=[detail_out])
    g_cross.select(click_cross,     inputs=[click_map_state], outputs=[detail_out])
    g_semantic.select(click_semantic, inputs=[click_map_state], outputs=[detail_out])
    g_color.select(click_color,     inputs=[click_map_state], outputs=[detail_out])

if __name__ == "__main__":
    demo.launch()