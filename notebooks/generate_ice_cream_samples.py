import sys

import numpy as np
import csv


def generate_ice_cream_data(num_samples=600):
    """
    Generates a dataset for the Ice Cream Sales problem.

    Args:
        num_samples (int): The number of data points to generate.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The feature matrix (Temperature, Is_Weekend).
            - y (np.ndarray): The target vector (Ice_Creams_Sold).
    """
    # for reproducibility
    np.random.seed(42)

    # 1. Generate Features (X)
    # Generate random temperatures between 10°C and 40°C
    temperatures = np.random.uniform(10, 40, num_samples)

    # Generate Is_Weekend flag (0 for weekday, 1 for weekend)
    # Let's assume weekends are about 2/7 of the days
    is_weekend = np.random.choice([0, 1], num_samples, p=[5 / 7, 2 / 7])

    # Combine features into a single matrix
    X = np.stack([temperatures, is_weekend], axis=1)

    # 2. Generate Target (y) using the hidden formula
    # Formula: Sales = (Temp * 8) + (Is_Weekend * 50) + Noise
    base_sales = (X[:, 0] * 8) + (X[:, 1] * 50)

    # Add some random "noise" to make it more realistic
    noise = np.random.normal(
        0, 15, num_samples
    )  # Gaussian noise with mean 0, std dev 15

    y = base_sales + noise

    # Ensure sales are not negative (you can't sell negative ice cream)
    y = np.maximum(y, 0)

    # Round to whole numbers for sales
    y = np.round(y)

    return X, y


def main():
    X, y = generate_ice_cream_data()

    with open("ice_cream_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Temperature", "Is_Weekend", "Ice_Creams_Sold"])
        for features, target in zip(X, y):
            writer.writerow([features[0], int(features[1]), int(target)])


if __name__ == "__main__":
    sys.exit(main())  # next section explains the use of sys.exit
