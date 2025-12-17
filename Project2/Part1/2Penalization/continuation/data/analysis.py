from math import log
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

report_params = {
    # Text Sizes (smaller, readable in print)
    'font.size': 18,            # General default
    'axes.labelsize': 18,       # x and y labels
    'axes.titlesize': 18,       # Title
    'xtick.labelsize': 16,      # Tick numbers
    'ytick.labelsize': 16,
    'legend.fontsize': 16,      # Legend text

    # Line & Marker Geometries (moderate for print)
    'lines.linewidth': 2.0,     # Thicker than default for clarity
    'lines.markersize': 4.0,    # Moderate marker size
    'lines.markeredgewidth': 0,  # Remove marker outline for cleaner look

    # Structural Geometries
    'axes.linewidth': 1.5,       # Thicker spines
    'xtick.major.width': 1.5,    # Thicker ticks
    'ytick.major.width': 1.5,
    'xtick.major.size': 6.0,     # Tick length
    'ytick.major.size': 6.0,

    # Fonts
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # Standard report font

    # Figure layout
    'figure.autolayout': True,
    'figure.figsize': (8, 6),     # Smaller for print/report
}

plt.rcParams.update(report_params)


df = pd.read_csv("continuation_clean.csv", sep=",")


idx = df["index"][:404].to_numpy()
penalty = df["Penalty"][:404].to_numpy()
densityLow = df["Density Low%"][:404].to_numpy()
densityMid = df["Density Mid%"][:404].to_numpy()
densityHigh = df["Density High%"][:404].to_numpy()
CMax = df["Compliance/MAX"][:404].to_numpy()

penaltyChange = np.where(penalty[1:] != penalty[:-1])[0] + 1
CMaxConverged = CMax[penaltyChange-1]
CMaxConverged = np.append(CMaxConverged, CMax[-1])

plt.plot(idx, densityLow, label="Density Low")
plt.plot(idx, densityMid, label="Density Mid")
plt.plot(idx, densityHigh, label="Density High")


plt.vlines(penaltyChange, ymin=0, ymax=60, colors='r',
           linestyles="--", linewidth=1, label="penalty increasess")

plt.xlabel("iteration")
plt.ylabel("Density %")

plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)
plt.legend(loc='upper right', frameon=True)

plt.show()


plt.plot(idx, CMax, label="")
plt.vlines(penaltyChange, ymin=0, ymax=np.max(CMax), colors='r',
           linestyles="--", linewidth=1, label="penalty increasess")

plt.xlabel("iteration")
plt.ylabel("Compliance")

plt.grid(True, which='both', axis='y',
         linestyle='--', linewidth=0.6, alpha=0.5)

plt.show()

penaltyValues = np.array([i/2 for i in range(2, 17)]
                         )


def model(x, a, b, c):
    return a * (1 - np.exp(-b*x)) + c


popt, pcov = curve_fit(model, penaltyValues, CMaxConverged)
x_fit = np.linspace(min(penaltyValues), max(penaltyValues), 200)

plt.scatter(penaltyValues, CMaxConverged)


plt.plot(x_fit, model(x_fit, *popt), 'r-', label='Fit')

plt.show()
