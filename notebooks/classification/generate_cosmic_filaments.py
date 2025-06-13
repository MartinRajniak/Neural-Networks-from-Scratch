import numpy as np
import csv
import sys


def generate_cosmic_filament_data(points_per_class=250, noise=0.2):
    """
    Generates a non-linear "Two Spirals" classification dataset.

    Args:
        points_per_class (int): The number of data points for each spiral arm.
        noise (float): The amount of random noise to add to the point coordinates.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The feature matrix (x_coordinate, y_coordinate).
            - y (np.ndarray): The target vector (class label 0 or 1).
    """
    # for reproducibility
    np.random.seed(42)

    X = np.zeros((points_per_class * 2, 2))
    y = np.zeros(points_per_class * 2, dtype="uint8")

    for i in range(points_per_class):
        # --- First Spiral (Class 0) ---
        radius = i / points_per_class * 5
        angle = i / points_per_class * 2.5 * np.pi

        # Add some randomness to the angle and radius
        random_angle = np.random.randn() * 0.1
        random_radius = np.random.randn() * 0.05

        x1 = (radius + random_radius) * np.sin(angle + random_angle)
        y1 = (radius + random_radius) * np.cos(angle + random_angle)

        X[i] = np.array([x1, y1]) + np.random.randn(2) * noise
        y[i] = 0

        # --- Second Spiral (Class 1) ---
        # Same logic, but we rotate it by 180 degrees (pi radians)
        x2 = -(radius + random_radius) * np.sin(angle + random_angle)
        y2 = -(radius + random_radius) * np.cos(angle + random_angle)

        X[i + points_per_class] = np.array([x2, y2]) + np.random.randn(2) * noise
        y[i + points_per_class] = 1

    return X, y


def main():
    X, y = generate_cosmic_filament_data(points_per_class=300, noise=0.2)

    with open("cosmic_filaments_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x_coord", "y_coord", "Spiral_Index"])
        for features, target in zip(X, y):
            writer.writerow([features[0], int(features[1]), int(target)])


if __name__ == "__main__":
    sys.exit(main())
