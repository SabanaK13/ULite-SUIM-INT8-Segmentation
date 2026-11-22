"""Exact binary saliency labels from SUIM paper arXiv:2004.01241v3.

Section V (PDF page 5); Figure 4 (page 4); Figure 6 (page 6).
HD / WR / RO / FV -> 1. BW / PF / RI / SR -> 0.
The upstream getSaliency() WR=0.8 behavior is intentionally not used.
"""
import numpy as np

# The index is 4*R_bit + 2*G_bit + B_bit, not an arbitrary class ID.
# Codes: BW, HD, PF, WR, RO, RI, FV, SR.
SALIENCY_LUT = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float32)


def rgb_to_saliency(mask_rgb):
    """Return float32 [H,W] labels from an original uint8 [H,W,3] RGB mask.

    Load masks with PIL.Image.open(...).convert('RGB'). OpenCV loads BGR:
    convert BGR to RGB first. Resize labels with nearest-neighbor only.
    Threshold each channel at 127.5, matching the authors' mask/255 > 0.5
    preprocessing. Official BMP masks can contain intermediate RGB values.
    Grayscale/index masks and separately encoded void labels are not supported.
    """
    mask = np.asarray(mask_rgb)
    if mask.ndim != 3 or mask.shape[-1] != 3 or mask.dtype != np.uint8:
        raise ValueError('Expected a uint8 RGB mask with shape [H,W,3].')
    bits = (mask > 127).astype(np.uint8)
    code = 4 * bits[..., 0] + 2 * bits[..., 1] + bits[..., 2]
    return SALIENCY_LUT[code]
