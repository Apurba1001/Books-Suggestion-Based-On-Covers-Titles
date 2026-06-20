# Project Decisions & Development Log
**Book Cover Recommendation System — Cross-Modal NLP & CV**

---

## Overview

This document records every significant technical decision made during the project, including the rationale, alternatives considered, and known limitations. It serves as the reference for the final presentation Q&A.

---

## 1. Problem Framing

**Decision:** Frame the task as *cross-modal retrieval* (find similar books given a cover image) rather than *classification* (predict the genre of a cover).

**Rationale:** Retrieval preserves the continuous similarity signal between books. Genre classification compresses that signal into a discrete label, losing the distinction between, say, a dark literary thriller and a pulp thriller — they share a genre label but a reader's interest in one doesn't imply interest in the other. Retrieval returns ranked results with similarity scores, which is directly useful to a reader.

**Connection to literature:** Toosi et al. (2025) demonstrated state-of-the-art results on book cover *classification*. Our work extends their findings to the *retrieval* setting, which is the natural next step toward a usable recommendation system.

---

## 2. Dataset Choice

**Decision:** Use the NYT Bestsellers dataset (~239 books across 10 genres, post-cleanup) as the primary evaluation dataset, supplemented with Google Books covers and manually curated ISBNs for underrepresented genres.

**Rationale:** The OpenLibrary dataset (387 books, original hackathon baseline) contains many obscure books that undermine demo quality — when a user uploads a Stephen King cover and gets back an unknown 1960s academic press book, the system looks broken even if the similarity score is technically correct. NYT bestsellers are universally recognizable, which makes evaluation results more interpretable and demos more convincing.

**Data pipeline:**
1. NYT Books API overview endpoint → top-15 per list, classified by description keywords
2. Google Books API → filled genre gaps using author-anchored queries + reference book filter
3. Manual ISBN list → guaranteed well-known books for horror, historical fiction, biography

**Known limitation:** NYT lists are biased toward contemporary English-language fiction and popular non-fiction. Literary diversity (translated fiction, independent publishers) is underrepresented.

**Genre distribution (final, post-cleanup):**

| Genre | Count | Primary source |
|---|---|---|
| horror | 27 | Manual (King, Stoker, Shelley) |
| science_fiction | 24 | NYT + Google Books (Herbert, Asimov) |
| fantasy | 24 | NYT + Google Books (Sanderson, Hobb) |
| romance | 23 | NYT + Google Books |
| mystery | 24 | Google Books (Christie, Connelly) |
| thriller | 24 | Google Books (Lee Child, Patterson) |
| historical_fiction | 29 | Google Books + Manual |
| biography | 23 | Google Books (Isaacson) + Manual |
| non-fiction | 23 | NYT |
| literary_fiction | 18 | NYT |

**Data cleanup (post-hackathon):**
Removed 17 entries from the original 256-book dataset:
- 14 audiobook editions — NYT lists include `audio-fiction` and `audio-nonfiction` lists alongside print lists, resulting in the same book appearing twice with different ISBNs and sometimes misclassified into different genres (e.g., "Theo of Golden" appeared as both literary_fiction and thriller)
- 1 duplicate print edition — The Final Target had two print ISBNs from different NYT lists
- 2 low-quality Victorian memoir compilations with decorative-border-only cover images

**Known issue identified during cleanup:** Google Books API results are biased by the requester's IP location. Our Austrian IP caused German-language academic texts to appear in English-restricted queries. Mitigated with the `lr=lang_en` parameter and a reference book filter, but some low-quality entries persisted and required manual removal.

**Known issue: broken/placeholder covers.** The Viscount Who Loved Me and Missing You both downloaded as audiobook placeholder images with near-identical pixel distributions, producing identical retrieval results across all four tracks regardless of query. These were identified by their anomalous behavior during evaluation and excluded from the golden set.

---

## 3. Four-Track Retrieval Architecture

**Decision:** Return recommendations across four separate tracks rather than a single blended ranking.

| Track | Signal | Method |
|---|---|---|
| Visual similarity | Cover art, style, composition | CLIP image → image cosine similarity |
| Cross-modal | Text on cover → visual match | CLIP text (from OCR) → image cosine similarity |
| Semantic / theme | Title, author, description meaning | SentenceTransformer text → text cosine similarity |
| Color palette | Dominant colors | k-means (k=5) palette → cosine similarity |

**Rationale:** Different tracks answer different user questions. A reader might want books that *look* similar (visual), books about *similar themes* (semantic), or books with a *similar aesthetic* (color). Fusing everything into one score hides which signal is driving the recommendation. Keeping tracks separate also enables ablation — we can measure each track's contribution independently, which is academically useful.

---

## 4. Model Selection (Baseline)

### 4.1 Visual encoder — CLIP ViT-B/32

**Decision:** Use `openai/clip-vit-base-patch32` via HuggingFace Transformers.

**Rationale:** CLIP's joint image-text embedding space enables the cross-modal track — the same model that encodes images also encodes text, and both live in a shared 512-dim space. This is the key architectural property that makes cross-modal retrieval possible without a separate alignment step.

**Why ViT-B/32 specifically:** Smallest CLIP variant that fits comfortably within both the laptop GPU (6GB VRAM RTX 3060 mobile) and the remote cluster allocation (~9.6GB VRAM, 1/5 L20). Larger variants (ViT-B/16, ViT-L/14) offer better quality but exceed our memory budget during concurrent inference.

**ViT-B/32 vs ViT-B/16 — patch size explained:** Both have the identical transformer backbone (12 layers, 768 hidden dimensions, 12 attention heads, "Base" size). The only difference is how the input image is divided into patches before entering the transformer:
- ViT-B/**32** splits a 224×224 image into 32×32 pixel patches → 7×7 = 49 tokens
- ViT-B/**16** splits a 224×224 image into 16×16 pixel patches → 14×14 = 196 tokens

Smaller patches give the model higher effective resolution — it can distinguish fine details like small author names, decorative borders, and subtle illustration textures that 32-pixel patches smear into a single token. The tradeoff is 4× more tokens to process, meaning slower inference and higher memory usage. Both produce the same 512-dim output vector, so nothing downstream changes.

**Alternatives considered:**
- SciCLIP — trained on scientific figures, worse for book covers
- SigLIP — stronger retrieval benchmark scores but too large for our hardware
- ResNet features — no cross-modal capability, purely visual

**Planned experiment:** Compare ViT-B/32 against ViT-B/16 (same memory footprint, smaller patch size = finer visual detail).

### 4.2 Text encoder — all-MiniLM-L6-v2

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` for the semantic track.

**Rationale:** Purpose-built for semantic similarity retrieval. 384-dim output, ~90MB, runs in milliseconds. Consistently strong on MTEB retrieval benchmarks relative to model size. The same model used by Toosi et al. (2025), which makes our results directly comparable.

**Alternatives considered:**
- BGE-base — stronger but ~500MB, marginal benefit at this scale
- TF-IDF — bag-of-words baseline, loses semantic meaning, useful only as comparison

### 4.3 OCR — EasyOCR

**Decision:** Use EasyOCR with English language model for text extraction from covers.

**Rationale:** Handles varied font styles, stylised typography, and skewed text better than Tesseract on book covers. The lazy-loading singleton pattern means it only loads once per session.

**Known limitation:** Misreads decorative and handwritten fonts common in romance and fantasy covers. Example: "By" misread as "Bx" on a plain cloth-bound biography cover. This is the primary motivation for the VLM replacement experiment.

**Planned experiment:** Replace EasyOCR with Qwen2-VL 2B descriptions (offline preprocessing step) and measure improvement on the semantic and cross-modal tracks.

### 4.4 Color extraction — k-means (k=5)

**Decision:** Extract 5 dominant colors via k-means on resized (100×100) pixel RGB values, sorted by brightness, flattened to a 15-dim vector.

**Rationale:** Simple, fast, interpretable. Color is a weak genre signal (literature suggests ~61% classification ceiling) but captures aesthetic similarity useful for cover design matching.

**Known limitation:** Color palette alone has near-zero P@3 across all golden set queries (0.10 mean). Useful as one signal among several but insufficient alone.

---

## 5. Evaluation Framework

**Decision:** Use Precision@3 (P@3) on a manually curated golden set of 10 queries.

**Rationale:** P@3 measures whether the system returns at least one good result in the top 3 — the most user-relevant threshold for a recommendation UI. The golden set is curated from actual retrieval results (relevance judging) rather than genre labels alone, which gives more honest scores.

**Golden set construction:** Initial expected matches based on genre labels gave near-zero scores because visual similarity doesn't respect genre boundaries. The golden set was rebuilt by running retrieval, inspecting top-5 results, and manually marking genuinely good matches. This is standard practice in information retrieval evaluation.

**Two queries dropped from golden set:**
- *The Viscount Who Loved Me* — cover downloaded as audio book placeholder, identical pixel distribution to *Missing You*, both producing identical retrieval results across all tracks. Replaced with *The Innovators* and *Ship of Magic*.

**Baseline P@3 results (ViT-B/32 + EasyOCR + MiniLM-L6-v2, post-cleanup):**

| Track | Mean P@3 |
|---|---|
| Visual | 0.37 |
| Semantic | 0.40 |
| Cross-modal | 0.30 |
| Color | 0.10 |

*Note: Pre-cleanup scores were slightly higher (Visual 0.47, Semantic 0.43, Cross-modal 0.33) because audiobook editions reuse the same cover art as print editions, creating artificially "free" matches in the index. The post-cleanup numbers reflect genuine retrieval quality on deduplicated data.*

---

## 6. Infrastructure Decisions

**Decision:** Use `uv` as the package manager and always run scripts as `uv run python`.

**Rationale:** Consistent virtual environment across Windows laptop and Linux cluster. Running `python3` directly bypasses the venv and causes `ModuleNotFoundError` on every dependency.

**Decision:** Store all embedding vectors as `.npy` files with a companion `row_index.json` mapping row numbers to book metadata.

**Rationale:** Simple, fast, no database dependency. Cosine similarity on 256×512 matrices is instantaneous with NumPy. Scales to ~10,000 books before needing an approximate nearest-neighbor index (Faiss, pgvector).

**Decision:** Separate embedding directories per dataset (`embeddings/` for OpenLibrary, `embeddings_nyt/` for NYT).

**Rationale:** Keeps datasets independent so experiments on one don't require rebuilding the other. The `--embeddings` CLI argument in `build_embeddings.py` and `evaluate.py` makes switching explicit.

**Known bug fixed:** `index.json` paths written with Windows backslashes (`\`) failed on the Linux cluster. Fixed with a one-line normalization script replacing `\\` with `/`. All JSON file operations now use `encoding="utf-8"` explicitly.

**Known bug fixed:** HuggingFace CLIP `get_image_features()` returned `BaseModelOutputWithPooling` instead of a tensor on the cluster's `transformers` version. Fixed by calling `model.vision_model()` and `model.visual_projection()` directly.

---

## 7. Planned Experiments (Semester)

### Comparison matrix

Two axes: visual encoder (rows) × text enrichment method (columns). Each cell is one configuration evaluated on the same golden set with P@3.

|  | EasyOCR | Qwen2-VL 2B descriptions |
|---|---|---|
| CLIP ViT-B/32 (OpenAI) | ✅ Baseline | E2 |
| CLIP ViT-B/16 (OpenAI) | E1 | E3 |
| OpenCLIP ViT-B/32 (LAION-2B) | E4 | E5 |

### Experiment details

| Experiment | What changes | What stays the same | Research question |
|---|---|---|---|
| E1: ViT-B/16 | Visual encoder patch size (32→16) | OCR text, sentence encoder | Does finer visual resolution improve cover retrieval? |
| E2: Qwen2-VL | OCR replaced by VLM descriptions | ViT-B/32 visual encoder | Does richer text improve semantic and cross-modal tracks? |
| E3: ViT-B/16 + Qwen2-VL | Both visual and text improved | Sentence encoder, color | Do improvements compound? |
| E4: OpenCLIP | Training data (400M→2B pairs) | Patch size (32), OCR text | Does more diverse training data help for book covers? |
| E5: OpenCLIP + Qwen2-VL | Best visual + best text | Sentence encoder, color | Best achievable configuration within our hardware budget |

### Why these experiments

- **E1 vs baseline** isolates the effect of visual resolution
- **E2 vs baseline** isolates the effect of text enrichment (extends Toosi et al. 2025 from classification to retrieval)
- **E4 vs baseline** isolates the effect of training data scale (same architecture, different pretraining corpus)
- **E3 and E5** test whether improvements are additive or redundant

### Why not SciCLIP

SciCLIP is trained on scientific figures (charts, microscopy, diagrams). Book covers contain photography, illustration, and decorative typography — a completely different visual domain. Using SciCLIP would test domain mismatch rather than model quality, which is not a useful comparison.

Each experiment changes one variable. All use the same golden set and P@3 metric so results are directly comparable.

---


## 8. Experiment Results

*All results on the final clean 182-book dataset after removing 66 Google Books placeholder covers (9103-byte "image not available" thumbnails) and 19 sub-15KB low-quality covers.*

### Final results table

| Configuration | Visual P@3 | Cross-modal P@3 | Semantic P@3 | Color P@3 |
|---|---|---|---|---|
| **Baseline:** ViT-B/32 + OCR | 0.43 | 0.40 | **0.37** | 0.10 |
| **E1:** ViT-B/16 + OCR | 0.43 | **0.43** | **0.37** | 0.10 |
| **E2:** ViT-B/32 + Qwen2-VL | 0.43 | 0.40 | 0.33 ↓ | 0.10 |
| **E3:** ViT-B/16 + Qwen2-VL | 0.43 | **0.43** | 0.33 ↓ | 0.10 |
| **E4:** OpenCLIP ViT-B/32 + OCR | **0.47** | 0.40 | **0.37** | 0.10 |
| **E5:** OpenCLIP ViT-B/32 + Qwen2-VL | **0.47** | 0.40 | 0.33 ↓ | 0.10 |
| **E6:** SigLIP 2 base-patch16 + OCR | 0.43 | 0.07 ↓↓ | **0.37** | 0.10 |

### Best configuration per track

| Track | Best config | P@3 | Why |
|---|---|---|---|
| Visual | OpenCLIP (E4/E5) | 0.47 | LAION-2B training improves image embedding space |
| Cross-modal | ViT-B/16 (E1/E3) | 0.43 | Finer patches resolve typographic detail for text→image matching |
| Semantic | Any OCR config | 0.37 | Sparse precise text (title + author) beats verbose VLM descriptions |
| Color | All configs | 0.10 | Hard ceiling — color palette does not correlate with book similarity |

### Data quality impact

The single most impactful change was removing placeholder covers, not switching models:

| Change | Visual Δ | Cross-modal Δ |
|---|---|---|
| Data cleanup (removing 85 junk covers) | **+0.06** | **+0.10** |
| Best model change (OpenCLIP) | +0.04 | 0 |
| Best architecture change (ViT-B/16) | 0 | +0.03 |

### E1: ViT-B/16 + EasyOCR

**Change:** Swapped CLIP visual encoder from ViT-B/32 (32×32 patches, 49 tokens) to ViT-B/16 (16×16 patches, 196 tokens).

**Result:** Cross-modal improved from 0.40 to 0.43 (+8%). All other tracks unchanged.

**Interpretation:** Finer patches better resolve typographic details on covers, improving text-to-image matching. The visual track (image→image) was unaffected — the top-3 visually similar covers are the same at both patch resolutions.

**Implementation note:** Query-time encoder must match index encoder. B/32 queries against B/16 embeddings produce near-zero scores (~0.07) despite identical dimensionality (512).

### E2: ViT-B/32 + Qwen2-VL 2B Descriptions

**Change:** Replaced EasyOCR with Qwen2-VL 2B generated descriptions (offline preprocessing).

**Result:** Semantic dropped from 0.37 to 0.33 (-11%). No improvement on any track.

**Interpretation:** On clean data, VLM descriptions provide no retrieval benefit. OCR extracts "THE SHINING Stephen King" — sparse but exact. The sentence transformer matches this against "IT Stephen King" via shared author name. VLM descriptions like "dark horror cover with menacing imagery" are richer but more generic, diluting precision.

**Key finding:** Contradicts Toosi et al. (2025) where VLM improved classification by +10 points. Classification benefits from rich discriminative features; retrieval benefits from sparse precise signals. Different tasks, different text requirements.

### E3: ViT-B/16 + Qwen2-VL 2B Descriptions

**Change:** Combined ViT-B/16 with VLM descriptions.

**Result:** Cross-modal 0.43 (same as E1 without VLM). Semantic 0.33 (same VLM penalty as E2).

**Interpretation:** On clean data, improvements are not additive. B/16 already captures the fine visual detail that VLM text was trying to match against — adding VLM contributes nothing extra. The semantic penalty persists regardless of visual encoder.

### E4: OpenCLIP ViT-B/32 + EasyOCR

**Change:** Swapped OpenAI CLIP (WIT, ~400M pairs) for OpenCLIP (LAION-2B, ~2B pairs). Same architecture, 5× more training data.

**Result:** Visual improved from 0.43 to 0.47 (+9%). All other tracks unchanged.

**Interpretation:** LAION-2B training genuinely improves image embeddings for book covers. This effect was completely hidden on dirty data (both scored 0.37 with placeholders) and only became visible after data cleanup. A critical lesson: measuring model differences requires clean evaluation data.

### E5: OpenCLIP ViT-B/32 + Qwen2-VL

**Change:** Combined OpenCLIP with VLM descriptions.

**Result:** Visual 0.47 (same as E4). Semantic 0.33 (same VLM penalty). Cross-modal unchanged.

**Interpretation:** Confirms two independent effects: OpenCLIP's visual advantage is text-source-independent, VLM's semantic penalty is visual-encoder-independent.

### E6: SigLIP 2 base-patch16-224 + EasyOCR

**Change:** Replaced CLIP with SigLIP 2 (sigmoid loss instead of softmax contrastive loss).

**Result:** Visual 0.43 (same as baseline). Cross-modal collapsed to 0.07 (-82%).

**Interpretation on clean data:** SigLIP 2 does NOT improve visual retrieval — its dirty-data advantage (0.50) was an artifact of placeholder images. The cross-modal collapse remains real: SigLIP's sigmoid loss creates weaker text-to-image alignment than CLIP's softmax contrastive loss, making sparse OCR fragments nearly useless for image retrieval.

**Key lesson:** SigLIP 2 is unsuitable for cross-modal book cover retrieval despite strong academic benchmark scores, which use well-formed captions rather than sparse OCR text.

### Color track investigation

Replaced baseline color extraction (RGB, k=5, brightness-sorted, 15-dim) with an improved version: LAB color space, cluster weights, border cropping, dominant-first sorting (20-dim).

**Result:** 0.10 P@3 — identical to baseline. Confirms color palette does not correlate with book similarity. Supported by literature: "Judging a Book by its Cover's Color" (2022).

### Summary of findings

| What we changed | What improved | What degraded | What didn't change |
|---|---|---|---|
| Data cleanup (85 junk covers) | Visual +0.06, Cross-modal +0.10 | — | Semantic, color |
| Finer patches (ViT-B/16) | Cross-modal +0.03 | — | Visual, semantic, color |
| VLM descriptions (Qwen2-VL) | — | Semantic -0.04 | Visual, cross-modal, color |
| More training data (OpenCLIP) | Visual +0.04 | — | Cross-modal, semantic, color |
| Different loss (SigLIP 2) | — | Cross-modal -0.33 | Visual, semantic, color |
| Improved color extraction (LAB) | — | — | Everything |

---

## 9. Conclusions

1. **Data quality dominates model choice.** Removing 85 placeholder covers improved retrieval more than any model swap. This is the single most important finding.

2. **No single model wins all tracks.** OpenCLIP is best for visual similarity (0.47). ViT-B/16 is best for cross-modal (0.43). OCR-based configs are best for semantic (0.37). The optimal system depends on which track the user values most.

3. **VLM descriptions hurt retrieval.** Contrary to Toosi et al. (2025) where VLM improved classification by +10 points, VLM descriptions consistently degrade semantic retrieval by -0.04. Classification and retrieval are fundamentally different tasks — classification benefits from rich discriminative features, retrieval benefits from sparse precise signals.

4. **Training objective matters more than training data.** SigLIP 2's sigmoid loss destroyed cross-modal retrieval (-82%) while OpenCLIP's larger dataset provided a modest visual improvement (+9%). The loss function shapes the embedding space geometry; data volume just fills it.

5. **Color palette is a dead end for book retrieval.** Two different extraction approaches (RGB + LAB) and literature review all converge: dominant colors don't discriminate between books. The 0.10 ceiling is fundamental, not technical.

6. **Dirty data masks real differences.** OpenCLIP's visual advantage and SigLIP 2's lack of visual advantage were both invisible on dirty data. Model comparison requires clean evaluation data to produce trustworthy conclusions.

---

## 10. Open Questions

- Would a hybrid text approach (OCR title+author concatenated with VLM mood description) preserve the precise signal while adding thematic context?
- Would fine-tuning CLIP's projection head on book cover/genre pairs improve visual track beyond OpenCLIP's 0.47?
- At what dataset size does cosine search over `.npy` files become too slow and require Faiss?
- Could a re-ranking step that combines multiple track scores outperform individual tracks?
