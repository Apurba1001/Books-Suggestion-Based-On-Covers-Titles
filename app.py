"""
app.py
------
Gradio web interface for the book cover recommendation system.
Upload a cover image and get top-3 recommendations per track,
displayed as a grid of thumbnails with titles and scores.

Usage:
    python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from PIL import Image
import numpy as np

from src.features.clip_encoder import encode_image, encode_text
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder import encode as encode_sentence
from src.search.retrieval      import load_index, search

TOP_K = 3

TRACK_LABELS = {
    "visual":      "🎨  Visual similarity",
    "cross_modal": "🔀  Cross-modal (text → image)",
    "semantic":    "📖  Semantic / theme",
    "color":       "🎨  Color palette",
}

# Load index once at startup
load_index()


# ── Core pipeline ──────────────────────────────────────────────────────────────

def recommend(image: np.ndarray):
    """
    Takes a numpy image array from Gradio, runs the full pipeline,
    and returns formatted outputs for each track tab.
    """
    if image is None:
        empty = gr.update(value=[], visible=False)
        return empty, empty, empty, empty, "Upload a cover image to get started."

    # Save temp file so our encoders can read it
    tmp_path = Path("tmp_query.jpg")
    Image.fromarray(image).save(tmp_path)

    # Feature extraction
    ocr_text     = extract_text(str(tmp_path))
    text         = ocr_text.strip() if ocr_text.strip() else "book cover"
    clip_img_vec = encode_image(str(tmp_path))
    clip_txt_vec = encode_text(text)
    sentence_vec = encode_sentence(text)
    color_vec    = extract_palette(str(tmp_path))

    results = search(
        clip_image_vec = clip_img_vec,
        clip_text_vec  = clip_txt_vec,
        sentence_vec   = sentence_vec,
        color_vec      = color_vec,
        k              = TOP_K,
    )

    tmp_path.unlink(missing_ok=True)

    ocr_display = f"**OCR detected:** {ocr_text}" if ocr_text.strip() else "**OCR:** nothing detected — using visual signal only"

    # Build gallery data per track: list of (image_path, caption) tuples
    def build_gallery(track):
        hits = results.get(track, [])
        gallery = []
        for hit in hits:
            caption = (
                f"#{hit['rank']}  {hit['title'][:40]}\n"
                f"Score: {hit['score']:.3f}  |  {hit['genre']}"
            )
            gallery.append((hit["filename"], caption))
        return gallery

    return (
        build_gallery("visual"),
        build_gallery("cross_modal"),
        build_gallery("semantic"),
        build_gallery("color"),
        ocr_display,
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Book Cover Recommender", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📚 Book Cover Recommendation System")
    gr.Markdown(
        "Upload a book cover (front only, or front and back) to get recommendations "
        "across four different similarity dimensions."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload cover image",
                type="numpy",
                height=400,
            )
            run_btn = gr.Button("Find similar books", variant="primary")
            ocr_out = gr.Markdown("OCR output will appear here after upload.")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🖼️  Visual similarity"):
                    visual_gallery = gr.Gallery(
                        label="Similar cover style & imagery",
                        columns=3, height=320, object_fit="cover"
                    )
                with gr.Tab("🔀  Cross-modal"):
                    cross_gallery = gr.Gallery(
                        label="OCR text → image search",
                        columns=3, height=320, object_fit="cover"
                    )
                with gr.Tab("📖  Semantic"):
                    semantic_gallery = gr.Gallery(
                        label="Similar theme & description",
                        columns=3, height=320, object_fit="cover"
                    )
                with gr.Tab("🎨  Color palette"):
                    color_gallery = gr.Gallery(
                        label="Similar dominant colors",
                        columns=3, height=320, object_fit="cover"
                    )

    run_btn.click(
        fn=recommend,
        inputs=[image_input],
        outputs=[visual_gallery, cross_gallery, semantic_gallery, color_gallery, ocr_out],
    )

    gr.Markdown(
        "---\n*Built with CLIP, SentenceTransformers, EasyOCR, and OpenLibrary data.*"
    )

if __name__ == "__main__":
    demo.launch()