#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid display issues
import matplotlib.pyplot as plt
import csv

PATH = "/ros2_ws/space/src/least_squares_fitter/scripts"
# Given data (in cm)
abs_dist = np.array([120, 100, 95, 90, 89, 85, 55, 50, 40])  # Absolute distance (y)
depth = np.array([249, 242, 236, 229, 228, 210, 191, 185, 170])  # Relative distance (x)

def main():
    # Fit a quadratic curve (y = ax^2 + bx + c)
    coefficients = np.polyfit(depth, abs_dist, 2)
    a, b, c = coefficients

    # Print the coefficients
    print(f"Quadratic equation: y = {a:.6f}x^2 + {b:.6f}x + {c:.6f}")

    # Save coefficients to a CSV file
    coefficients_filename = os.path.join(PATH, "quadratic_coefficients.csv")
    with open(coefficients_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["a", "b", "c"])
        writer.writerow([a, b, c])

    print(f"Quadratic coefficients saved to {coefficients_filename}")

    # Plot the curve
    x_vals = np.linspace(min(depth) - 10, max(depth) + 10, 100)
    y_vals = a * x_vals**2 + b * x_vals + c

    plt.scatter(depth, abs_dist, color='red', label='Data Points')
    plt.plot(x_vals, y_vals, color='blue', label='Fitted Quadratic Curve')
    plt.xlabel('Relative Distance (Depth) in cm')
    plt.ylabel('Absolute Distance in cm')
    plt.title('Quadratic Fit by Least Squares Method')
    plt.legend()
    plt.grid()

    # Save plot
    plt.savefig(os.path.join(PATH, 'quadratic_fit.png'))
    plt.close()

if __name__=="__main__":
    main()

