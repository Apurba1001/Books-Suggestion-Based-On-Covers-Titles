"""
ui.py
-----
Simple UI to view recommendation results with cover images.
Click any result to see its full index metadata.

Usage:
    uv run python main/ui.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import numpy as np

from src.features.clip_encoder import encode_image, encode_text
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder  import encode as encode_sentence
from src.search.retrieval       import load_index, search

TOP_K = 3

load_index()

# Load full index for metadata lookup
INDEX_PATH = Path("data/index_updated.json")
if INDEX_PATH.exists():
    with open(INDEX_PATH, encoding="utf-8") as f:
        FULL_INDEX = {book["cover_id"]: book for book in json.load(f)}
else:
    with open("data/index.json", encoding="utf-8") as f:
        FULL_INDEX = {book["cover_id"]: book for book in json.load(f)}


def list_covers():
    covers_dir = Path("data/covers")
    if covers_dir.exists():
        return sorted([f.name for f in covers_dir.glob("*.jpg")])
    return []


def recommend(filename: str):
    if not filename:
        return None, "", [], [], [], [], {}, ""

    path = Path("data/covers") / filename
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

    # Store all results keyed by "track:index" for click lookup
    click_map = {}

    def build_gallery(track):
        hits = results.get(track, [])
        gallery = []
        for i, hit in enumerate(hits):
            authors = ", ".join(hit["authors"]) if hit["authors"] else "Unknown"
            caption = (
                f"#{hit['rank']}  {hit['title']}\n"
                f"Score: {hit['score']:.3f} | {hit['genre']}\n"
                f"by {authors}"
            )
            gallery.append((hit["filename"], caption))
            click_map[f"{track}:{i}"] = hit["cover_id"]
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


def on_gallery_click(track: str, click_map: dict, evt: gr.SelectData):
    key = f"{track}:{evt.index}"
    cover_id = click_map.get(key)
    if cover_id is None:
        return "No metadata found."

    book = FULL_INDEX.get(cover_id)
    if book is None:
        return f"cover_id `{cover_id}` not found in index."

    lines = [f"### 📋 Index entry for: {book.get('title', '?')}"]
    lines.append("```json")
    lines.append(json.dumps(book, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)


with gr.Blocks(title="Book Cover Recommender", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📚 Book Cover Recommender")
    gr.Markdown("*Click any result image to see its full index metadata below.*")

    click_map_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            dropdown = gr.Dropdown(
                choices=list_covers(),
                label="Select a cover",
                allow_custom_value=True,
            )
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

    run_btn.click(
        fn=recommend,
        inputs=[dropdown],
        outputs=[query_img, ocr_out, g_visual, g_cross, g_semantic, g_color, click_map_state, detail_out],
    )
    
    def click_visual(click_map, evt: gr.SelectData):
        return on_gallery_click("visual", click_map, evt)

    def click_cross(click_map, evt: gr.SelectData):
        return on_gallery_click("cross_modal", click_map, evt)

    def click_semantic(click_map, evt: gr.SelectData):
        return on_gallery_click("semantic", click_map, evt)

    def click_color(click_map, evt: gr.SelectData):
        return on_gallery_click("color", click_map, evt)

    g_visual.select(click_visual, inputs=[click_map_state], outputs=[detail_out])
    g_cross.select(click_cross, inputs=[click_map_state], outputs=[detail_out])
    g_semantic.select(click_semantic, inputs=[click_map_state], outputs=[detail_out])
    g_color.select(click_color, inputs=[click_map_state], outputs=[detail_out])

if __name__ == "__main__":
    demo.launch()