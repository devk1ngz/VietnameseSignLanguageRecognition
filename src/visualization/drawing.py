"""
visualization/drawing.py
------------------------
Lightweight helper utilities for drawing overlaid text and annotations
onto OpenCV / NumPy image frames during real-time inference.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: tuple = (20, 20),
    color: tuple = (255, 255, 255),
    font_size: int = 20,
    background_color: tuple = None,
    font_path: str = None,
) -> np.ndarray:
    """
    Draw anti-aliased text onto a BGR NumPy image (as returned by OpenCV).

    Uses Pillow for rendering so that Unicode / Vietnamese characters are
    handled correctly, then converts the result back to a BGR NumPy array
    for continued use with OpenCV.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (H, W, 3) as returned by ``cv2.imread`` /
        ``cv2.VideoCapture.read``.
    text : str
        The string to render.  Supports Unicode / multi-byte characters.
    position : tuple[int, int]
        (x, y) pixel coordinate of the *top-left* corner of the text block.
    color : tuple[int, int, int]
        Text colour in **RGB** order (not BGR), e.g. ``(255, 255, 255)``
        for white.
    font_size : int
        Point size of the rendered font.
    background_color : tuple[int, int, int] | None
        Optional background rectangle colour in RGB order.  When *None* no
        background is drawn.
    font_path : str | None
        Absolute path to a ``.ttf`` / ``.otf`` font file.  Falls back to
        Pillow's built-in bitmap font when *None*.

    Returns
    -------
    np.ndarray
        A copy of *image* with the text composited onto it (BGR, uint8).
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a non-None NumPy ndarray")

    # Convert BGR → RGB so Pillow renders colours correctly
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Load font
    try:
        if font_path is not None:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Attempt to load a system font; fall back gracefully
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    x, y = position

    # Optional background rectangle
    if background_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        padding = 4
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding,
             bbox[2] + padding, bbox[3] + padding],
            fill=background_color,
        )

    draw.text((x, y), text, fill=color, font=font)

    # Convert back to BGR for OpenCV
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result
