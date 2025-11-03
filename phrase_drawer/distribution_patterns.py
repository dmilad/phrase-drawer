"""
Distribution-Based Pattern Generators

This module provides functions for creating patterns based on image
brightness distributions, enabling density-based pattern generation.
"""

from PIL import Image
import numpy as np
from .edge_orientation import compute_orientation_field


def create_brightness_distribution(image_path, chunk_size=5, compute_orientations=True,
                                   orientation_smoothing=2.0, orientation_method='structure_tensor',
                                   integration_sigma=2.0):
    """
    Create a brightness distribution from a grayscale image by dividing it into chunks.

    Args:
        image_path: Path to the image file
        chunk_size: Size of chunks to divide the image into (e.g., 5 means 5x5 pixel chunks)
        compute_orientations: If True, also compute edge orientations (default: True)
        orientation_smoothing: Smoothing sigma for orientation field (default: 2.0)
        orientation_method: Method for computing orientations (default: 'structure_tensor')
                           Options: 'structure_tensor', 'sobel'
        integration_sigma: Integration sigma for structure tensor method (default: 2.0)

    Returns:
        Tuple of (weights, chunk_centers, orientations) where:
            - weights: numpy array of normalized brightness values (sum to 1.0)
            - chunk_centers: list of (x, y) tuples representing chunk center positions
            - orientations: numpy array of edge orientation angles in degrees (None if compute_orientations=False)
    """
    # Load image and convert to grayscale
    img = Image.open(image_path)

    # Handle transparency by compositing onto white background
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image on the background using the alpha channel as mask
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        img = background

    # Now convert to grayscale
    img = img.convert('L')
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

    # Compute edge orientations if requested
    orientations = None
    if compute_orientations:
        orientations, _ = compute_orientation_field(
            image_path, chunk_size, orientation_smoothing,
            method=orientation_method, integration_sigma=integration_sigma
        )

    return weights, chunk_centers, orientations


def sample_positions_from_distribution(weights, chunk_centers, canvas_width, canvas_height,
                                       border, num_samples, add_jitter=False,
                                       invert_distribution=False, orientations=None,
                                       orientation_jitter=5):
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
        orientations: numpy array of orientation angles in degrees (optional)
        orientation_jitter: Random jitter to add to orientations in degrees (default: 5)

    Returns:
        If orientations is None: List of (x, y) tuples in canvas coordinates
        If orientations provided: List of (x, y, angle) tuples in canvas coordinates
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

    # Calculate aspect-ratio-preserving scale
    # Scale to fit the canvas while maintaining aspect ratio
    src_aspect = src_width / src_height if src_height > 0 else 1.0
    canvas_aspect = inner_width / inner_height if inner_height > 0 else 1.0

    if src_aspect > canvas_aspect:
        # Source is wider - fit to width
        scale = inner_width / src_width if src_width > 0 else 1.0
        scaled_width = inner_width
        scaled_height = src_height * scale
        offset_x = 0
        offset_y = (inner_height - scaled_height) / 2
    else:
        # Source is taller or same - fit to height
        scale = inner_height / src_height if src_height > 0 else 1.0
        scaled_width = src_width * scale
        scaled_height = inner_height
        offset_x = (inner_width - scaled_width) / 2
        offset_y = 0

    # Sample indices from the distribution
    sampled_indices = np.random.choice(len(chunk_centers), size=num_samples, p=sampling_weights)

    positions = []
    for idx in sampled_indices:
        src_x, src_y = chunk_centers[idx]

        # Map from source image coordinates to canvas coordinates
        # Apply uniform scale and centering offset
        canvas_x = border + offset_x + (src_x - min_x) * scale
        canvas_y = border + offset_y + (src_y - min_y) * scale

        # Add jitter if requested
        if add_jitter:
            jitter_amount = min(inner_width, inner_height) * 0.01  # 1% of smaller dimension
            canvas_x += np.random.uniform(-jitter_amount, jitter_amount)
            canvas_y += np.random.uniform(-jitter_amount, jitter_amount)

            # Clamp to inner bounds
            canvas_x = max(border, min(canvas_width - border, canvas_x))
            canvas_y = max(border, min(canvas_height - border, canvas_y))

        # Include orientation if available
        if orientations is not None:
            angle = orientations[idx]
            # Add jitter to orientation for natural appearance
            if orientation_jitter > 0:
                angle += np.random.uniform(-orientation_jitter, orientation_jitter)
            positions.append((int(canvas_x), int(canvas_y), angle))
        else:
            positions.append((int(canvas_x), int(canvas_y)))

    return positions


def generate_sampled_pattern(image_path, canvas_width=6400, canvas_height=4800,
                             border=400, chunk_size=5, iterations=50,
                             samples_per_iteration=10, add_jitter=False,
                             invert_distribution=False, compute_orientations=True,
                             orientation_smoothing=2.0, orientation_jitter=5,
                             orientation_method='structure_tensor', integration_sigma=2.0):
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
        compute_orientations: If True, compute and return edge orientations (default: True)
        orientation_smoothing: Smoothing sigma for orientation field (default: 2.0)
        orientation_jitter: Random jitter to add to orientations in degrees (default: 5)
        orientation_method: Method for computing orientations (default: 'structure_tensor')
                           Options: 'structure_tensor', 'sobel'
        integration_sigma: Integration sigma for structure tensor method (default: 2.0)

    Returns:
        If compute_orientations=True: List of (x, y, angle) tuples
        If compute_orientations=False: List of (x, y) tuples
    """
    # Create brightness distribution (and optionally orientations)
    weights, chunk_centers, orientations = create_brightness_distribution(
        image_path, chunk_size, compute_orientations, orientation_smoothing,
        orientation_method, integration_sigma
    )

    all_positions = []

    # Perform iterative sampling
    for _ in range(iterations):
        positions = sample_positions_from_distribution(
            weights, chunk_centers, canvas_width, canvas_height, border,
            samples_per_iteration, add_jitter, invert_distribution,
            orientations, orientation_jitter
        )
        all_positions.extend(positions)

    return all_positions
