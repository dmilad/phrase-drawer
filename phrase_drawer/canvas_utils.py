"""
Canvas and Image Utilities

This module provides functions for creating canvases, loading images,
and creating sample test images.
"""

from PIL import Image, ImageDraw


def create_canvas(width=1024, height=1024, color='white'):
    """
    Create a blank canvas.

    Args:
        width: Canvas width in pixels
        height: Canvas height in pixels
        color: Background color (default: white)

    Returns:
        PIL Image object
    """
    return Image.new('RGB', (width, height), color)


def load_small_image(path):
    """
    Load a small image with transparency support.

    Args:
        path: Path to the image file

    Returns:
        PIL Image object in RGBA mode
    """
    img = Image.open(path)
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    return img


def remove_background(image_path, output_path=None):
    """
    Remove the background from an image using AI-based segmentation.

    This function uses the rembg library (U2-Net model) to automatically
    detect and remove the background from an image, leaving only the
    foreground subject with transparency.

    Args:
        image_path: Path to the input image file
        output_path: Optional path to save the result. If None, doesn't save.

    Returns:
        PIL Image object in RGBA mode with transparent background

    Example:
        >>> # Remove background and save
        >>> result = remove_background('person.jpg', 'person_no_bg.png')
        >>>
        >>> # Remove background without saving
        >>> result = remove_background('person.jpg')

    Note:
        On first run, this will download the U2-Net model (~176MB).
        Requires the 'rembg' package to be installed.
    """
    from rembg import remove

    # Load the input image
    input_image = Image.open(image_path)

    # Remove background - returns RGBA image with transparent background
    output_image = remove(input_image)

    # Save if output path is provided
    if output_path:
        output_image.save(output_path)

    return output_image


def create_sample_small_image(width=10, height=20, shape='circle', color=(0, 0, 0, 255)):
    """
    Create a sample small image with transparency for testing.

    Args:
        width: Image width
        height: Image height
        shape: 'circle', 'square', or 'triangle'
        color: RGBA color tuple

    Returns:
        PIL Image object with transparency
    """
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if shape == 'circle':
        # Draw a circle/ellipse
        draw.ellipse([0, 0, width-1, height-1], fill=color)
    elif shape == 'square':
        # Draw a square
        draw.rectangle([0, 0, width-1, height-1], fill=color)
    elif shape == 'triangle':
        # Draw a triangle
        points = [(width//2, 0), (width-1, height-1), (0, height-1)]
        draw.polygon(points, fill=color)

    return img
