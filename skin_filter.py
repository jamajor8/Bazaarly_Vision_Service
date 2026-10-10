"""
Removes hands and forearms from a garment mask, so the cutout contains the garment and not the person holding it.
Put this file next to vision_service.py.
"""
import cv2
import numpy as np

# Skin range in YCrCb (Y, Cr, Cb). Works for many skin tones in normal light, less well in very dim or
# colored lighting. If hands are still included, widen Cr/Cb a little; if garments get eaten, narrow them.
SKIN_LO = (0, 133, 77)
SKIN_HI = (255, 180, 135)

MAX_SKIN_SHARE = 0.6    # if more than 60% of the mask is "skin", the garment itself is skin-toned: skip filtering
MIN_KEEP_SHARE = 0.25   # if filtering would leave less than 25% of the mask, assume it misfired: skip filtering


def remove_skin(rgb: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """
    rgb:    HxWx3 uint8 image in RGB order (the downscaled frame the mask was made from)
    binary: HxW uint8 mask of 0/1 (foreground from the segmentation model)
    Returns a cleaned 0/1 mask of the same size.
    """
    total = int(binary.sum())
    if total == 0:
        return binary

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, SKIN_LO, SKIN_HI)
    # close small gaps (shadows between fingers) so each hand becomes one region
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    skin_in_mask = (skin > 0) & (binary > 0)
    if skin_in_mask.sum() > MAX_SKIN_SHARE * total:
        return binary   # skin-colored garment (tan, peach...): leave the mask alone

    # only remove sizeable skin regions (hands, arms). Tiny skin-like specks on a garment are left alone.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((skin > 0).astype(np.uint8))
    min_area = 0.004 * binary.size
    big_skin = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            big_skin[labels == i] = 1
    big_skin = cv2.dilate(big_skin, np.ones((3, 3), np.uint8), iterations=1)   # take the fringe too

    cleaned = binary.copy()
    cleaned[big_skin > 0] = 0

    # cut thin leftovers (a strip of sleeve still attached to the garment, finger shadows)
    k = max(3, int(round(binary.shape[1] * 0.02)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    if cleaned.sum() < MIN_KEEP_SHARE * total:
        return binary   # filter removed almost everything: don't trust it
    return cleaned