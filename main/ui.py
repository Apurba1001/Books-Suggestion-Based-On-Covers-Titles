"""
ui.py
-----
Professional Gradio UI for the book cover recommendation system.

Usage:
    uv run python ui.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import numpy as np

from src.features.clip_encoder import encode_image, encode_text, set_model as set_clip_model
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder  import encode as encode_sentence
from src.search.retrieval       import load_index, search

TOP_K = 3

# ── Config ─────────────────────────────────────────────────────────────────────

MODELS = [
    ("CLIP ViT-B/32 + OCR (baseline)",    "openai/clip-vit-base-patch32",           "embeddings_nyt"),
    ("CLIP ViT-B/16 + OCR",               "openai/clip-vit-base-patch16",           "embeddings_nyt_b16"),
    ("CLIP ViT-B/32 + Qwen2-VL",          "openai/clip-vit-base-patch32",           "embeddings_nyt_vlm"),
    ("CLIP ViT-B/16 + Qwen2-VL",          "openai/clip-vit-base-patch16",           "embeddings_nyt_b16_vlm"),
    ("OpenCLIP ViT-B/32 + OCR",           "openclip:ViT-B-32/laion2b_s34b_b79k",   "embeddings_nyt_openclip"),
    ("SigLIP 2 base-patch16 + OCR",       "siglip:google/siglip2-base-patch16-224", "embeddings_nyt_siglip2"),
]

COVERS_DIR  = "data/covers_nyt"
INDEX_FILES = ("data/nyt_index_updated.json", "data/nyt_index.json")

GENRE_COLORS = {
    "horror": "#dc2626", "science_fiction": "#2563eb", "fantasy": "#7c3aed",
    "romance": "#ec4899", "mystery": "#0891b2", "thriller": "#d97706",
    "historical_fiction": "#059669", "biography": "#4f46e5",
    "non-fiction": "#0d9488", "literary_fiction": "#6366f1",
}

CUSTOM_CSS = """
.header-title {
    font-size: 2.2em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}
.header-subtitle {
    font-size: 1em !important;
    color: var(--neutral-500) !important;
    margin-top: 0 !important;
}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_available_models():
    return [label for label, _, emb_dir in MODELS if Path(emb_dir).exists()]

def get_model_config(label):
    for l, clip_id, emb_dir in MODELS:
        if l == label:
            return clip_id, emb_dir
    return None, None

def load_full_index():
    for p in INDEX_FILES:
        if Path(p).exists():
            with open(p, encoding="utf-8") as f:
                books = json.load(f)
            return {str(b.get("isbn") or b.get("cover_id", "")): b for b in books}
    return {}

def list_covers():
    d = Path(COVERS_DIR)
    return sorted([f.name for f in d.glob("*.jpg")]) if d.exists() else []

def score_color(score):
    if score >= 0.7:   return "#22c55e"
    elif score >= 0.4: return "#eab308"
    else:              return "#ef4444"


# ── State ──────────────────────────────────────────────────────────────────────

current_full_index  = load_full_index()
current_model_label = None


# ── Core logic ─────────────────────────────────────────────────────────────────

def switch_model(model_label):
    global current_model_label
    clip_id, emb_dir = get_model_config(model_label)
    if not clip_id:
        return f"⚠ Model not found"
    set_clip_model(clip_id)
    load_index(emb_dir)
    current_model_label = model_label
    return f"✅ **{model_label}**"


def build_gallery(results, track):
    """Build list of (filepath, caption) tuples for gr.Gallery, skipping missing/corrupt files."""
    from PIL import Image as PILImage
    hits = results.get(track, [])
    gallery = []
    rank = 0
    for hit in hits:
        abs_path = str(Path(hit["filename"]).resolve())
        if not Path(abs_path).exists():
            continue
        try:
            img = PILImage.open(abs_path)
            img.verify()
        except Exception:
            continue
        rank += 1
        authors = ", ".join(hit["authors"]) if hit["authors"] else "Unknown"
        genre   = hit["genre"].replace("_", " ")
        caption = (
            f"#{rank}  {hit['title']}\n"
            f"Score: {hit['score']:.3f}  •  {genre}\n"
            f"by {authors}"
        )
        gallery.append((abs_path, caption))
        if len(gallery) >= TOP_K:
            break
    return gallery


def recommend(model_label, filename, uploaded_img):
    if not model_label:
        return None, "", [], [], [], [], {}

    global current_model_label
    if model_label != current_model_label:
        switch_model(model_label)

    # Determine image source
    if uploaded_img is not None:
        img_path = uploaded_img
        book_id  = None
        source   = "📤 Uploaded image"
    elif filename:
        img_path = str(Path(COVERS_DIR) / filename)
        book_id  = Path(filename).stem.split("_", 1)[-1]
        source   = "📁 Dataset"
    else:
        return None, "Select a cover or upload an image.", [], [], [], [], {}

    if not Path(img_path).exists():
        return None, f"⚠ File not found: {img_path}", [], [], [], [], {}

    # Feature extraction
    ocr_text     = extract_text(str(img_path))
    text         = ocr_text.strip() if ocr_text.strip() else Path(img_path).stem.replace("_", " ")
    clip_img_vec = encode_image(str(img_path))
    clip_txt_vec = encode_text(text)
    sentence_vec = encode_sentence(text)
    color_vec    = extract_palette(str(img_path))

    results = search(
        clip_image_vec = clip_img_vec,
        clip_text_vec  = clip_txt_vec,
        sentence_vec   = sentence_vec,
        color_vec      = color_vec,
        k              = TOP_K + 5,  # fetch extra to compensate for missing files
        exclude_id     = book_id,
    )

    # Info panel
    ocr_display = ocr_text if ocr_text.strip() else "*nothing detected*"
    info = f"**{source}** · {model_label}\n\n**OCR:** {ocr_display}"

    return (
        str(img_path),
        info,
        build_gallery(results, "visual"),
        build_gallery(results, "cross_modal"),
        build_gallery(results, "semantic"),
        build_gallery(results, "color"),
        {},
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

available_models = get_available_models()

with gr.Blocks(title="Book Cover Recommender") as demo:

    gr.Markdown("# 📚 Book Cover Recommender", elem_classes="header-title")
    gr.Markdown(
        "Upload any book cover or select from the dataset — compare results across models and similarity tracks.",
        elem_classes="header-subtitle"
    )

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            with gr.Group():
                model_dd = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="🧠 Model",
                )
                model_status = gr.Markdown("")

            with gr.Group():
                cover_dd = gr.Dropdown(
                    choices=list_covers(),
                    label="📁 Select from dataset",
                    allow_custom_value=True,
                )
                upload_img = gr.Image(
                    label="📤 Or upload a cover",
                    type="filepath",
                    height=160,
                )

            run_btn = gr.Button("🔍 Find Similar Books", variant="primary", size="lg")

            query_img = gr.Image(label="Query Cover", height=240)
            info_out  = gr.Markdown("")

        # ── Right panel ───────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=500):
            with gr.Tabs():
                with gr.Tab("🖼️ Visual"):
                    g_visual = gr.Gallery(
                        label="Similar cover style & imagery",
                        columns=3, rows=1, height=380,
                        object_fit="contain", show_label=True,
                    )
                with gr.Tab("🔀 Cross-modal"):
                    g_cross = gr.Gallery(
                        label="OCR text → image search",
                        columns=3, rows=1, height=380,
                        object_fit="contain", show_label=True,
                    )
                with gr.Tab("📖 Semantic"):
                    g_semantic = gr.Gallery(
                        label="Similar theme & description",
                        columns=3, rows=1, height=380,
                        object_fit="contain", show_label=True,
                    )
                with gr.Tab("🎨 Color"):
                    g_color = gr.Gallery(
                        label="Similar color palette",
                        columns=3, rows=1, height=380,
                        object_fit="contain", show_label=True,
                    )

    # ── Events ─────────────────────────────────────────────────────────────

    #demo.load(fn=switch_model, inputs=[model_dd], outputs=[model_status])
    model_dd.change(fn=switch_model, inputs=[model_dd], outputs=[model_status])

    run_btn.click(
        fn=recommend,
        inputs=[model_dd, cover_dd, upload_img],
        outputs=[query_img, info_out, g_visual, g_cross, g_semantic, g_color, gr.State({})],
    )

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(Path(COVERS_DIR).resolve())],
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    )