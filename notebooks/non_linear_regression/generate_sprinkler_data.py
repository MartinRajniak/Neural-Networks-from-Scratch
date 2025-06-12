import numpy as np
import csv
import sys


def generate_sprinkler_data(num_samples=800):
    """
    Generates a non-linear dataset modeling a smart sprinkler's
    watering pattern on a square field.

    Args:
        num_samples (int): The number of data points to generate.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The feature matrix (x_coordinate, y_coordinate).
            - y (np.ndarray): The target vector (Water_Level).
    """
    # for reproducibility
    np.random.seed(42)

    # Field coordinates from -10 to 10
    x_coords = np.random.uniform(-10, 10, num_samples)
    y_coords = np.random.uniform(-10, 10, num_samples)

    # Combine into a single feature matrix
    X = np.stack([x_coords, y_coords], axis=1)

    # --- The Non-Linear Function (2D Gaussian) ---
    # Sprinkler is at the center (0, 0)
    center_x, center_y = 0, 0

    # Peak water level is 100
    peak_intensity = 100

    # How wide the water spreads
    spread = 5.0

    # Calculate the distance squared from the center
    dist_sq = (X[:, 0] - center_x) ** 2 + (X[:, 1] - center_y) ** 2

    # The Gaussian function to determine the base water level
    base_water_level = peak_intensity * np.exp(-dist_sq / (2 * spread**2))

    # Add some random "noise" (e.g., from wind, uneven ground)
    noise = np.random.normal(0, 3, num_samples)

    y = base_water_level + noise

    # Ensure water level doesn't go below zero
    y = np.maximum(y, 0)

    return X, y


def main():
    X, y = generate_sprinkler_data(800)

    with open("smart_sprinkler_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x_coord", "y_coord", "Water_Level"])
        for features, target in zip(X, y):
            writer.writerow([features[0], int(features[1]), int(target)])


if __name__ == "__main__":
    sys.exit(main())
