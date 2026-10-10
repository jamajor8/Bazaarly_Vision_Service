"""
Asks the vision model where the garment is (ignoring the person holding it) and returns a bounding box.
Put this file next to vision_service.py. The cutout then only happens inside this box.
"""
import time
from typing import Optional

from google.genai import types
from PIL import Image
from pydantic import BaseModel

BOX_MIN_INTERVAL = 5.0   # seconds between lookups. Protects the free-tier rate limit: 12 calls/min at most.
_last_call = 0.0

PROMPT = (
    "Find the single garment or clothing item that is being held up or shown to the camera. "
    "Do NOT include the person holding it, their hands, arms, face, body, or the clothes they are wearing. "
    "Return JSON of the form {\"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates scaled 0 to 1000, "
    "drawn tightly around the garment only. If no garment is being shown, return {\"box_2d\": null}."
)


class _Box(BaseModel):
    box_2d: Optional[list[int]] = None


def find_garment_box(client, model: str, img: Image.Image):
    """
    Returns (status, box). box is (x0, y0, x1, y1) as 0-1 fractions of the frame, or None.
    status: "ok"        a box was found
            "none"      the model says no garment is being shown
            "throttled" called too recently; skip this frame
            "failed"    the call or the parsing failed; caller should fall back to no box
    """
    global _last_call
    now = time.time()
    if now - _last_call < BOX_MIN_INTERVAL:
        return "throttled", None
    _last_call = now

    small = img.copy()
    small.thumbnail((768, 768))
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[small, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_Box,
                max_output_tokens=1000,   # headroom in case the model spends tokens thinking
            ),
        )
        box = _Box.model_validate_json(resp.text).box_2d
    except Exception as e:
        print("find_garment_box failed:", e)
        return "failed", None

    if box is None:
        return "none", None
    if len(box) != 4:
        return "failed", None
    y0, x0, y1, x1 = [min(1000, max(0, v)) / 1000 for v in box]
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return "failed", None
    return "ok", (x0, y0, x1, y1)