#!/bin/bash
# cleanup.sh
# Run once from project root to reorganize the codebase.
# Safe to re-run — uses -n flag on mv to avoid overwriting.

set -e
echo "── Reorganizing project structure ──"

# 1. Move build_embeddings.py from embeddings/ to src/features/
if [ -f "embeddings/build_embeddings.py" ]; then
    mv embeddings/build_embeddings.py src/features/build_embeddings.py
    echo "  ✓ moved embeddings/build_embeddings.py → src/features/"
fi

# 2. Move main/main.py to root
if [ -f "main/main.py" ]; then
    mv main/main.py main.py
    echo "  ✓ moved main/main.py → main.py"
fi

# 3. Move main/ui.py to root
if [ -f "main/ui.py" ]; then
    mv main/ui.py ui.py
    echo "  ✓ moved main/ui.py → ui.py"
fi

# 4. Remove app.py (duplicate of ui.py)
if [ -f "app.py" ]; then
    rm app.py
    echo "  ✓ removed app.py (duplicate)"
fi

# 5. Remove empty index.py
if [ -f "src/search/index.py" ]; then
    rm src/search/index.py
    echo "  ✓ removed src/search/index.py (empty)"
fi

# 6. Remove now-empty main/ folder
if [ -d "main" ] && [ -z "$(ls -A main)" ]; then
    rmdir main
    echo "  ✓ removed empty main/ folder"
fi

# 7. Fix The Martian genre in nyt_index.json
uv run python - << 'PYEOF'
import json
from pathlib import Path

path = Path("data/nyt_index.json")
with open(path, encoding="utf-8") as f:
    index = json.load(f)

fixed = 0
for book in index:
    if book.get("isbn") == "9798217300556" and book.get("genre") == "fantasy":
        book["genre"] = "science_fiction"
        fixed += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

print(f"  ✓ fixed {fixed} genre mislabel (The Martian: fantasy → science_fiction)")
PYEOF

# 8. Flag broken covers
uv run python - << 'PYEOF'
import numpy as np
from PIL import Image
from pathlib import Path

suspects = [
    ("data/covers_nyt/romance_9780063279469.jpg",   "The Viscount Who Loved Me"),
    ("data/covers_nyt/thriller_9781471282645.jpg",   "Missing You"),
]
for path, title in suspects:
    if Path(path).exists():
        img = np.array(Image.open(path))
        std = img.std()
        status = "⚠ likely placeholder" if std < 10 else "✓ looks ok"
        print(f"  {status}  {title}  (pixel std={std:.1f})")
    else:
        print(f"  ✗ file missing: {title}")
PYEOF

echo ""
echo "✓ Cleanup complete."
echo ""
echo "Next: update sys.path in main.py, ui.py, evaluate.py, visualize.py"
echo "      (parents[1] is now correct for all root-level scripts)"