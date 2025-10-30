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


def create_brightness_distribution(image_path, chunk_size=5):
    """
    Create a brightness distribution from a grayscale image by dividing it into chunks.

    Args:
        image_path: Path to the image file
        chunk_size: Size of chunks to divide the image into (e.g., 5 means 5x5 pixel chunks)

    Returns:
        Tuple of (weights, chunk_centers) where:
            - weights: numpy array of normalized brightness values (sum to 1.0)
            - chunk_centers: list of (x, y) tuples representing chunk center positions
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    height, width = img_array.shape

    # Calculate number of chunks in each dimension
    num_chunks_y = height // chunk_size
    num_chunks_x = width // chunk_size

    weights = []
    chunk_centers = []

    # Iterate through chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            # Get chunk boundaries
            y_start = i * chunk_size
            y_end = (i + 1) * chunk_size
            x_start = j * chunk_size
            x_end = (j + 1) * chunk_size

            # Extract chunk and calculate average brightness
            chunk = img_array[y_start:y_end, x_start:x_end]
            avg_brightness = np.mean(chunk)

            # Store brightness as weight
            weights.append(avg_brightness)

            # Store chunk center position (in original image coordinates)
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            chunk_centers.append((center_x, center_y))

    # Convert to numpy array and normalize to create probability distribution
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    return weights, chunk_centers


def sample_positions_from_distribution(weights, chunk_centers, canvas_width, canvas_height,
                                       border, num_samples, add_jitter=False,
                                       invert_distribution=False):
    """
    Sample positions from a brightness distribution, mapping them to canvas coordinates.

    Args:
        weights: numpy array of normalized probability weights
        chunk_centers: list of (x, y) tuples in source image coordinates
        canvas_width: Width of the target canvas
        canvas_height: Height of the target canvas
        border: Border size (positions will be constrained to inner area)
        num_samples: Number of positions to sample
        add_jitter: If True, add random noise to positions (default: False)
        invert_distribution: If True, invert weights so darker areas have higher probability

    Returns:
        List of (x, y) tuples in canvas coordinates
    """
    # Invert distribution if requested (darker = more likely)
    if invert_distribution:
        max_weight = np.max(weights)
        sampling_weights = max_weight - weights
        sampling_weights = sampling_weights / np.sum(sampling_weights)
    else:
        sampling_weights = weights

    # Calculate inner canvas area
    inner_width = canvas_width - 2 * border
    inner_height = canvas_height - 2 * border

    # Get source image dimensions from chunk centers
    if chunk_centers:
        max_x = max(x for x, y in chunk_centers)
        max_y = max(y for x, y in chunk_centers)
        min_x = min(x for x, y in chunk_centers)
        min_y = min(y for x, y in chunk_centers)
        src_width = max_x - min_x
        src_height = max_y - min_y
    else:
        return []

    # Sample indices from the distribution
    sampled_indices = np.random.choice(len(chunk_centers), size=num_samples, p=sampling_weights)

    positions = []
    for idx in sampled_indices:
        src_x, src_y = chunk_centers[idx]

        # Map from source image coordinates to canvas coordinates
        # Normalize to [0, 1] range first
        norm_x = (src_x - min_x) / src_width if src_width > 0 else 0.5
        norm_y = (src_y - min_y) / src_height if src_height > 0 else 0.5

        # Map to inner canvas area
        canvas_x = border + norm_x * inner_width
        canvas_y = border + norm_y * inner_height

        # Add jitter if requested
        if add_jitter:
            jitter_amount = min(inner_width, inner_height) * 0.01  # 1% of smaller dimension
            canvas_x += np.random.uniform(-jitter_amount, jitter_amount)
            canvas_y += np.random.uniform(-jitter_amount, jitter_amount)

            # Clamp to inner bounds
            canvas_x = max(border, min(canvas_width - border, canvas_x))
            canvas_y = max(border, min(canvas_height - border, canvas_y))

        positions.append((int(canvas_x), int(canvas_y)))

    return positions


def generate_sampled_pattern(image_path, canvas_width=6400, canvas_height=4800,
                             border=400, chunk_size=5, iterations=50,
                             samples_per_iteration=10, add_jitter=False,
                             invert_distribution=False):
    """
    Generate a pattern by iteratively sampling from a brightness distribution.

    This function creates a probability distribution based on the brightness values
    of an image divided into chunks, then samples positions from this distribution
    multiple times to create a density pattern.

    Args:
        image_path: Path to the source image for creating the distribution
        canvas_width: Width of the canvas (default: 6400)
        canvas_height: Height of the canvas (default: 4800)
        border: Border size around canvas (default: 400)
        chunk_size: Size of chunks for brightness sampling (default: 5)
        iterations: Number of sampling iterations (default: 50)
        samples_per_iteration: Number of samples per iteration (default: 10)
        add_jitter: If True, add random noise to sampled positions (default: False)
        invert_distribution: If True, darker areas get higher probability (default: False)

    Returns:
        List of (x, y) tuples representing all sampled positions
    """
    # Create brightness distribution
    weights, chunk_centers = create_brightness_distribution(image_path, chunk_size)

    all_positions = []

    # Perform iterative sampling
    for _ in range(iterations):
        positions = sample_positions_from_distribution(
            weights, chunk_centers, canvas_width, canvas_height, border,
            samples_per_iteration, add_jitter, invert_distribution
        )
        all_positions.extend(positions)

    return all_positions


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
