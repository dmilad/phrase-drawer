"""
Image Overlay Functionality

This module provides the core functionality for overlaying images
onto a canvas at specified positions.
"""
import random


def overlay_images(canvas, small_image, positions, center_images=True, rotation_range=0):
    """
    Overlay a small image at multiple positions on a canvas.

    Args:
        canvas: PIL Image object (the base canvas)
        small_image: PIL Image object (the image to overlay)
        positions: List of position tuples. Can be either:
                  - (x, y) tuples for placement (uses rotation_range for rotation)
                  - (x, y, angle) tuples with specific rotation angles (ignores rotation_range)
        center_images: If True, center the small image at each position (x, y).
                      If False, place top-left corner at (x, y). Default: True
        rotation_range: Maximum rotation angle in degrees (e.g., 15 means random rotation
                       between -15 and +15 degrees). Default is 0 (no rotation).
                       Ignored if positions include rotation angles.
                       Note: Rotation is applied around the image center before positioning.

    Returns:
        PIL Image object with overlays applied

    Note:
        Images are rotated counter-clockwise (PIL convention). When using with
        edge_orientation module, angles are automatically calculated to align
        images parallel to detected edges in the source image.
    """
    # Create a copy to avoid modifying the original
    result = canvas.copy()

    for position in positions:
        # Handle both (x, y) and (x, y, angle) tuple formats
        if len(position) == 3:
            # Position includes rotation angle
            x, y, angle = position
            use_specific_angle = True
        else:
            # Position is just (x, y)
            x, y = position
            use_specific_angle = False

        # Apply rotation
        # Note: PIL's rotate() rotates around the image center by default.
        # With expand=True, the rotated image is centered in a new bounding box
        # that fits the entire rotated content.
        if use_specific_angle:
            # Use the specific angle provided
            rotated_image = small_image.rotate(angle, expand=True, resample=3)
        elif rotation_range > 0:
            # Apply random rotation if specified
            angle = random.uniform(-rotation_range, rotation_range)
            rotated_image = small_image.rotate(angle, expand=True, resample=3)
        else:
            # No rotation
            rotated_image = small_image

        # Get dimensions of the (possibly rotated) image
        img_width, img_height = rotated_image.size

        if center_images:
            # Adjust position to center the image at (x, y)
            # Since PIL's rotate with expand=True centers the rotated content in the new box,
            # the center of the rotated image is at (img_width/2, img_height/2)
            # To place this center at position (x, y), we paste at (x - w/2, y - h/2)
            # Use round() instead of integer division for better accuracy
            paste_x = round(x - img_width / 2)
            paste_y = round(y - img_height / 2)
        else:
            paste_x = x
            paste_y = y

        # Paste with alpha channel as mask for transparency
        try:
            result.paste(rotated_image, (paste_x, paste_y), rotated_image)
        except:
            # If image goes off canvas, skip it
            pass

    return result
