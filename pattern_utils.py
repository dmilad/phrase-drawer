"""
Image Pattern Generator Utilities

This module provides functions for creating image patterns by overlaying
smaller images on a larger canvas in various arrangements.
"""

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


# ============================================================================
# Canvas and Image Utilities
# ============================================================================

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


def display_image(img, title='Image'):
    """
    Display an image using matplotlib.

    Args:
        img: PIL Image object
        title: Title for the plot
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# ============================================================================
# Pattern Generation Functions
# ============================================================================

def generate_grid_positions(canvas_width, canvas_height, spacing_x, spacing_y,
                           offset_x=0, offset_y=0):
    """
    Generate positions for a regular grid pattern.

    Args:
        canvas_width: Width of the canvas
        canvas_height: Height of the canvas
        spacing_x: Horizontal spacing between images
        spacing_y: Vertical spacing between images
        offset_x: Starting x offset
        offset_y: Starting y offset

    Returns:
        List of (x, y) tuples
    """
    positions = []
    y = offset_y
    while y < canvas_height:
        x = offset_x
        while x < canvas_width:
            positions.append((x, y))
            x += spacing_x
        y += spacing_y
    return positions


def generate_circle_positions(center_x, center_y, radius, num_points,
                             start_angle=0, rotation=0):
    """
    Generate positions along a circle.

    Args:
        center_x: Circle center x coordinate
        center_y: Circle center y coordinate
        radius: Circle radius
        num_points: Number of images to place
        start_angle: Starting angle in degrees (default: 0)
        rotation: Additional rotation in degrees (default: 0)

    Returns:
        List of (x, y) tuples
    """
    positions = []
    angle_step = 360 / num_points

    for i in range(num_points):
        angle = math.radians(start_angle + (i * angle_step) + rotation)
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        positions.append((x, y))

    return positions


def generate_concentric_circles_positions(center_x, center_y, min_radius,
                                         max_radius, num_circles, points_per_circle):
    """
    Generate positions for concentric circles pattern.

    Args:
        center_x: Center x coordinate
        center_y: Center y coordinate
        min_radius: Inner circle radius
        max_radius: Outer circle radius
        num_circles: Number of concentric circles
        points_per_circle: Number of points per circle

    Returns:
        List of (x, y) tuples
    """
    positions = []

    if num_circles == 1:
        radii = [min_radius]
    else:
        radii = np.linspace(min_radius, max_radius, num_circles)

    for i, radius in enumerate(radii):
        # Optional: rotate each circle slightly for visual interest
        rotation = i * 15  # 15 degrees per circle
        circle_positions = generate_circle_positions(
            center_x, center_y, radius, points_per_circle, rotation=rotation
        )
        positions.extend(circle_positions)

    return positions


def generate_spiral_positions(center_x, center_y, start_radius, end_radius,
                             num_points, num_rotations=3):
    """
    Generate positions along a spiral.

    Args:
        center_x: Spiral center x coordinate
        center_y: Spiral center y coordinate
        start_radius: Starting radius
        end_radius: Ending radius
        num_points: Number of images to place
        num_rotations: Number of full rotations (default: 3)

    Returns:
        List of (x, y) tuples
    """
    positions = []
    total_angle = num_rotations * 360

    for i in range(num_points):
        t = i / num_points
        radius = start_radius + (end_radius - start_radius) * t
        angle = math.radians(total_angle * t)

        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        positions.append((x, y))

    return positions


def generate_square_outline_positions(center_x, center_y, side_length, points_per_side):
    """
    Generate positions along the outline of a square.

    Args:
        center_x: Square center x coordinate
        center_y: Square center y coordinate
        side_length: Length of each side
        points_per_side: Number of points per side

    Returns:
        List of (x, y) tuples
    """
    positions = []
    half_side = side_length // 2

    # Top side
    for i in range(points_per_side):
        x = int(center_x - half_side + (side_length * i / points_per_side))
        y = center_y - half_side
        positions.append((x, y))

    # Right side
    for i in range(points_per_side):
        x = center_x + half_side
        y = int(center_y - half_side + (side_length * i / points_per_side))
        positions.append((x, y))

    # Bottom side
    for i in range(points_per_side):
        x = int(center_x + half_side - (side_length * i / points_per_side))
        y = center_y + half_side
        positions.append((x, y))

    # Left side
    for i in range(points_per_side):
        x = center_x - half_side
        y = int(center_y + half_side - (side_length * i / points_per_side))
        positions.append((x, y))

    return positions


# ============================================================================
# Image Overlay Function
# ============================================================================

def overlay_images(canvas, small_image, positions, center_images=True):
    """
    Overlay a small image at multiple positions on a canvas.

    Args:
        canvas: PIL Image object (the base canvas)
        small_image: PIL Image object (the image to overlay)
        positions: List of (x, y) tuples for placement
        center_images: If True, center the small image at each position.
                      If False, use top-left corner (default: True)

    Returns:
        PIL Image object with overlays applied
    """
    # Create a copy to avoid modifying the original
    result = canvas.copy()

    # Get small image dimensions
    img_width, img_height = small_image.size

    for x, y in positions:
        if center_images:
            # Adjust position to center the image
            paste_x = x - img_width // 2
            paste_y = y - img_height // 2
        else:
            paste_x = x
            paste_y = y

        # Paste with alpha channel as mask for transparency
        try:
            result.paste(small_image, (paste_x, paste_y), small_image)
        except:
            # If image goes off canvas, skip it
            pass

    return result


# ============================================================================
# Visualization Helper
# ============================================================================

def visualize_positions(canvas_width, canvas_height, positions, title='Pattern Positions'):
    """
    Visualize the pattern positions on a plot (useful for debugging).

    Args:
        canvas_width: Canvas width
        canvas_height: Canvas height
        positions: List of (x, y) tuples
        title: Plot title
    """
    plt.figure(figsize=(10, 10))

    if positions:
        x_coords, y_coords = zip(*positions)
        plt.scatter(x_coords, y_coords, c='blue', alpha=0.6, s=50)

    plt.xlim(0, canvas_width)
    plt.ylim(canvas_height, 0)  # Invert y-axis to match image coordinates
    plt.title(f"{title} ({len(positions)} positions)")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.show()
