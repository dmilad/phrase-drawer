"""
Phrase Drawer - Image Pattern Generator

A toolkit for creating artistic patterns by arranging images on a canvas
in various geometric and distribution-based patterns.
"""

# Canvas and Image utilities
from .canvas_utils import (
    create_canvas,
    load_small_image,
    create_sample_small_image,
    remove_background,
)

# Distribution-based pattern generators
from .distribution_patterns import (
    create_brightness_distribution,
    sample_positions_from_distribution,
    generate_sampled_pattern,
)

# Image overlay
from .overlay import overlay_images

# Visualization utilities
from .visualization import (
    display_image,
    visualize_positions,
)

__all__ = [
    # Canvas utilities
    'create_canvas',
    'load_small_image',
    'create_sample_small_image',
    'remove_background',
    # Geometric patterns
    'generate_grid_positions',
    'generate_circle_positions',
    'generate_concentric_circles_positions',
    'generate_spiral_positions',
    'generate_square_outline_positions',
    # Distribution patterns
    'create_brightness_distribution',
    'sample_positions_from_distribution',
    'generate_sampled_pattern',
    # Overlay
    'overlay_images',
    # Visualization
    'display_image',
    'visualize_positions',
]
