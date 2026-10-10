"""
Garment scanning service. Run:
    pip install fastapi uvicorn python-multipart google-genai pydantic pillow opencv-python-headless sentence-transformers
    export GEMINI_API_KEY=...          # optional: export GEMINI_MODEL=<model name from AI Studio>
    uvicorn vision_service:app --host 0.0.0.0 --port 8000

Endpoints (OpenAPI docs auto-generated at /docs for the other devs):
    POST /scan/frame           multipart: session_id, image  -> new_item | duplicate_pending | awaiting_user | suppressed | rejected
    PATCH /items/{id}          json: any of overview / full_description / sections{} / not_visible  (vendor edits the draft)
    POST /items/{id}/approve   draft -> approved, frees the session for the next item
    POST /items/{id}/discard   throws the draft away (can be rescanned)
    POST /items/{id}/retry     re-run the description after an error
    (While a draft is open, /scan/frame returns "awaiting_approval"; after approve/discard there is a short cooldown.)
    POST /scan/pending/{id}/resolve   json: {"action": "add" | "dismiss"}   (answer to a duplicate_pending)
    GET  /items/{id}/image/{n} -> the stored JPEG for a view (use it to show the existing item in the prompt)
    POST /items/{id}/views     multipart: image              -> adds a view (back, tag, inside) and re-describes
    GET  /items/{id}           -> status + full description
    GET  /items?session_id=    -> list items
"""
import base64, io, json, os, uuid, time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from google import genai
from google.genai import types
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Model name is a setting: if Google retires or renames it, change GEMINI_MODEL, not the code.
# Check AI Studio for the names available on your key's free tier.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
DATA = Path("data"); DATA.mkdir(exist_ok=True)
BLUR_MIN = 30.0          # Laplacian variance at 640px; below = too blurry. Tune: see "blur_score" in responses.
DUP_THRESHOLD = 0.92     # cosine similarity; tune with same-item vs similar-item pairs.

client = genai.Client()   # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment
embedder = SentenceTransformer("clip-ViT-B-32")
app = FastAPI(title="Garment Vision Service")

ITEMS: dict[str, dict] = {}
SESSION_EMB: dict[str, list[tuple[str, np.ndarray]]] = {}

# Duplicate-confirmation state
PENDING: dict[str, dict] = {}                 # pending_id -> flagged frame awaiting the vendor's answer
SESSION_PENDING: dict[str, str] = {}          # session_id -> pending_id (only one open question per session)
SUPPRESS: dict[tuple[str, str], float] = {}   # (session_id, item_id) -> time until we stop re-asking
SUPPRESS_SECONDS = 15                         # how long the same item can stay in view without re-prompting

# One-item-at-a-time review flow
ACTIVE: dict[str, str] = {}                   # session_id -> item_id of the draft awaiting approval
COOLDOWN: dict[str, float] = {}               # session_id -> time before the next scan is accepted
COOLDOWN_SECONDS = 3                          # gives the vendor time to put the item down


# ---------- schema ----------
class Sections(BaseModel):
    overall_shape_and_fit: str
    colors_and_dyeing: str
    fabric_and_texture: str
    neckline_collar: str
    sleeves_and_cuffs: str
    body_and_hem: str
    closures_and_hardware: str
    pockets: str
    stitching_and_seams: str
    prints_graphics_text: str = Field(description="Transcribe any visible text exactly")
    wear_flaws_and_marks: str = Field(description="Each flaw with its location and size")
    visible_labels_and_tags: str
    inside_and_lining: str


class Description(BaseModel):
    overview: str = Field(description="2-3 sentence summary")
    sections: Sections
    full_description: str = Field(description="Exhaustive ordered prose covering everything in the sections")
    not_visible: list[str] = Field(description="Parts of the garment not shown in any image")


PROMPT = """You are cataloguing ONE garment from the attached image(s) for a search index.
Write an exhaustive, purely observational description. Be as long and specific as possible.

Rules:
- Work region by region: overall shape/fit, colors, fabric appearance, neckline/collar, sleeves/cuffs,
  body, hem, closures, hardware, pockets, stitching, prints/graphics, visible text, wear/flaws, labels/tags, inside/lining.
- Give counts, positions (e.g. 'about 3 inches below the left shoulder seam'), relative sizes, pattern direction,
  thread colors, and exact colors in plain words.
- Transcribe any visible text exactly, in quotes.
- Describe flaws with location and approximate size.
- Observation only. No subjective or marketing words (vintage, stylish, high quality). Do NOT guess brand, era,
  fiber content, or value. Only mention a brand if it is literally printed on the garment, and say so as transcription.
- Ignore hands, hangers, background, and other garments.
- If a region is not visible, write 'Not visible' for that section and add it to not_visible. Never invent detail.
- If multiple images are given they are different views of the same garment; merge them into one description."""


# ---------- helpers ----------
register_heif_opener()   # lets Pillow open iPhone .heic photos


def to_pil(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the image. Send a JPEG, PNG or HEIC file.")


def blur_score(img: Image.Image) -> float:
    # Downscale first: Laplacian variance depends on resolution, so full-size phone photos
    # of sharp but plain garments score low. Measuring at a fixed size keeps the threshold meaningful.
    small = img.copy()
    small.thumbnail((640, 640))
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def b64(img: Image.Image, max_side: int = 1568) -> str:
    img = img.copy(); img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def embed(img: Image.Image) -> np.ndarray:
    v = embedder.encode(img, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def save_item(item: dict):
    (DATA / f"{item['item_id']}.json").write_text(json.dumps(item, indent=2))


def run_description(item_id: str):
    """Background job: call the VLM on all views of the item."""
    item = ITEMS[item_id]
    try:
        contents = []
        for i, v in enumerate(item["views"]):
            im = Image.open(v["path"]).convert("RGB")
            im.thumbnail((1568, 1568))
            contents.append(f"Image {i+1}:")
            contents.append(im)
        contents.append(PROMPT)
        t0 = time.time()
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Description,
                max_output_tokens=8000,
            ),
        )
        desc = Description.model_validate_json(resp.text)
        item["descriptions"] = desc.model_dump()
        item["descriptions_original"] = desc.model_dump()   # untouched copy; vendor edits go in "descriptions"
        item["status"] = "ready"                            # = draft, waiting for the vendor to review
        item["model_info"] = {"vlm": MODEL, "prompt_version": "v1",
                              "latency_s": round(time.time() - t0, 1),
                              "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", None),
                              "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", None)}
    except Exception as e:
        item["status"] = "error"; item["error"] = str(e)
    save_item(item)


def add_view(item: dict, img: Image.Image, label: str):
    n = len(item["views"])
    path = DATA / f"{item['item_id']}_{n}.jpg"
    img.save(path, "JPEG", quality=92)
    item["views"].append({"view_id": str(uuid.uuid4()), "path": str(path), "label": label})


def create_item(session_id: str, img: Image.Image, vec: np.ndarray,
                background: BackgroundTasks, duplicate_of: Optional[str] = None) -> dict:
    item_id = str(uuid.uuid4())
    item = {"schema_version": "2.0", "item_id": item_id, "session_id": session_id,
            "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "processing", "views": [], "descriptions": None,
            "descriptions_original": None, "edited": False,
            "possible_duplicate_of": duplicate_of}
    add_view(item, img, "front")
    ITEMS[item_id] = item
    ACTIVE[session_id] = item_id
    SESSION_EMB.setdefault(session_id, []).append((item_id, vec))
    save_item(item)
    background.add_task(run_description, item_id)
    return item


class Resolution(BaseModel):
    action: Literal["add", "dismiss"]   # add = real duplicate item, dismiss = mistake


# ---------- endpoints ----------
@app.post("/scan/frame")
async def scan_frame(background: BackgroundTasks, session_id: str = Form(...), image: UploadFile = File(...)):
    # One item at a time: ignore frames while a draft is being reviewed.
    if session_id in ACTIVE:
        return {"result": "awaiting_approval", "item_id": ACTIVE[session_id]}
    if COOLDOWN.get(session_id, 0) > time.time():
        return {"result": "cooldown"}

    img = to_pil(await image.read())

    score = blur_score(img)
    if score < BLUR_MIN:
        return {"result": "rejected", "reason": "blurry", "blur_score": score}

    vec = embed(img)
    best_id, best_sim = None, 0.0
    for item_id, other in SESSION_EMB.get(session_id, []):
        sim = float(vec @ other)
        if sim > best_sim:
            best_id, best_sim = item_id, sim
    if best_id and best_sim >= DUP_THRESHOLD:
        # A question is already open for this session: don't spawn another one.
        if session_id in SESSION_PENDING:
            return {"result": "awaiting_user", "pending_id": SESSION_PENDING[session_id]}
        # Vendor recently answered about this item and it's probably still in view.
        if SUPPRESS.get((session_id, best_id), 0) > time.time():
            return {"result": "suppressed", "possible_duplicate_of": best_id}

        pending_id = str(uuid.uuid4())
        path = DATA / f"pending_{pending_id}.jpg"
        img.save(path, "JPEG", quality=92)
        PENDING[pending_id] = {"session_id": session_id, "path": str(path),
                               "duplicate_of": best_id, "similarity": best_sim}
        SESSION_PENDING[session_id] = pending_id
        existing = ITEMS[best_id]
        return {"result": "duplicate_pending", "pending_id": pending_id,
                "similarity": best_sim,
                "existing_item": {
                    "item_id": best_id,
                    "image_url": f"/items/{best_id}/image/0",
                    "overview": (existing["descriptions"] or {}).get("overview")}}

    item = create_item(session_id, img, vec, background,
                       duplicate_of=best_id if best_sim > 0.8 else None)
    return {"result": "new_item", "item_id": item["item_id"], "status": "processing"}


@app.post("/scan/pending/{pending_id}/resolve")
def resolve_pending(pending_id: str, body: Resolution, background: BackgroundTasks):
    """Vendor's answer to 'this looks like an item you already scanned'."""
    p = PENDING.pop(pending_id, None)
    if not p:
        raise HTTPException(404, "pending duplicate not found (already resolved?)")
    session_id = p["session_id"]
    SESSION_PENDING.pop(session_id, None)
    until = time.time() + SUPPRESS_SECONDS
    SUPPRESS[(session_id, p["duplicate_of"])] = until

    if body.action == "dismiss":           # mistake: throw the frame away
        Path(p["path"]).unlink(missing_ok=True)
        return {"result": "dismissed"}

    img = Image.open(p["path"]).convert("RGB")   # real duplicate: log it as its own item
    Path(p["path"]).unlink(missing_ok=True)
    item = create_item(session_id, img, embed(img), background, duplicate_of=p["duplicate_of"])
    SUPPRESS[(session_id, item["item_id"])] = until   # the new item is still in view too
    return {"result": "new_item", "item_id": item["item_id"], "status": "processing",
            "possible_duplicate_of": p["duplicate_of"]}


class DescriptionEdit(BaseModel):
    """Any subset of fields. Sections are merged key by key."""
    overview: Optional[str] = None
    full_description: Optional[str] = None
    sections: Optional[dict[str, str]] = None
    not_visible: Optional[list[str]] = None


def finish_review(item: dict):
    """Free the session for the next item."""
    sid = item["session_id"]
    if ACTIVE.get(sid) == item["item_id"]:
        ACTIVE.pop(sid, None)
    COOLDOWN[sid] = time.time() + COOLDOWN_SECONDS


@app.patch("/items/{item_id}")
def edit_item(item_id: str, edit: DescriptionEdit):
    """Vendor corrects the draft description before approving."""
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "item not found")
    if item["status"] != "ready":
        raise HTTPException(409, f"item is '{item['status']}', only drafts that are ready can be edited")
    d = item["descriptions"]
    if edit.overview is not None:
        d["overview"] = edit.overview
    if edit.full_description is not None:
        d["full_description"] = edit.full_description
    if edit.not_visible is not None:
        d["not_visible"] = edit.not_visible
    if edit.sections:
        bad = set(edit.sections) - set(Sections.model_fields)
        if bad:
            raise HTTPException(422, f"unknown sections: {sorted(bad)}")
        d["sections"].update(edit.sections)
    item["edited"] = True
    save_item(item)
    return item


@app.post("/items/{item_id}/approve")
def approve_item(item_id: str):
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "item not found")
    if item["status"] != "ready":
        raise HTTPException(409, f"item is '{item['status']}', wait until it is 'ready'")
    item["status"] = "approved"
    item["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    SUPPRESS[(item["session_id"], item_id)] = time.time() + SUPPRESS_SECONDS  # still in view? don't re-prompt
    finish_review(item)
    save_item(item)
    return item


@app.post("/items/{item_id}/discard")
def discard_item(item_id: str):
    """Vendor rejects the draft (bad scan, wrong item). It can be rescanned."""
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "item not found")
    item["status"] = "discarded"
    sid = item["session_id"]
    SESSION_EMB[sid] = [(i, v) for i, v in SESSION_EMB.get(sid, []) if i != item_id]
    finish_review(item)
    save_item(item)
    return {"item_id": item_id, "status": "discarded"}


@app.post("/items/{item_id}/retry")
def retry_item(item_id: str, background: BackgroundTasks):
    """Re-run the description if it errored."""
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "item not found")
    if item["status"] != "error":
        raise HTTPException(409, "only errored items can be retried")
    item["status"] = "processing"
    background.add_task(run_description, item_id)
    return {"item_id": item_id, "status": "processing"}


@app.get("/items/{item_id}/image/{view_index}")
def get_item_image(item_id: str, view_index: int):
    item = ITEMS.get(item_id)
    if not item or view_index >= len(item["views"]):
        raise HTTPException(404, "image not found")
    return FileResponse(item["views"][view_index]["path"], media_type="image/jpeg")


@app.post("/items/{item_id}/views")
async def add_item_view(item_id: str, background: BackgroundTasks,
                        image: UploadFile = File(...), label: str = Form("extra")):
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "item not found")
    if item["status"] not in ("ready", "error") or item.get("edited"):
        # Re-describing would overwrite the vendor's edits, so extra views must come before editing.
        raise HTTPException(409, "add extra views before editing, and only while the item is a draft")
    add_view(item, to_pil(await image.read()), label)
    item["status"] = "processing"
    save_item(item)
    background.add_task(run_description, item_id)
    return {"item_id": item_id, "status": "processing", "num_views": len(item["views"])}


@app.get("/items/{item_id}")
def get_item(item_id: str):
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "item not found")
    return item


@app.get("/items")
def list_items(session_id: Optional[str] = None, status: Optional[str] = None):
    return [i for i in ITEMS.values()
            if (session_id is None or i["session_id"] == session_id)
            and (status is None or i["status"] == status)]