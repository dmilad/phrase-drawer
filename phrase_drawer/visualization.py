"""
Visualization Utilities

This module provides functions for displaying images and visualizing
pattern positions for debugging and analysis.
"""

import matplotlib.pyplot as plt


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
