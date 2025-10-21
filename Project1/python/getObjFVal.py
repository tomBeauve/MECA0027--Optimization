from math import sin, cos
from counters import counters
import numpy as np


def getObjFVal(x, functionID):
    if functionID == 1:
        fval = 4*x[0]**2 + 5*x[0]*x[1] + 3*x[1]**2 + 4*x[0] - 3*x[1] + 5
    elif functionID == 2:
        fval = 0.5*(x[0]**2 + x[1]**2) - 2 * cos(x[0]) * \
            x[1] - 10 * sin(x[1]) - 0.5 * x[0]*x[1]
    elif functionID == 3:
        fval = 0.1*(x[0]**2 + x[1]**2) - 0.3 * cos(x[0]) * \
            x[1] - 3 * sin(x[1]) - 0.1 * x[0]*x[1]

    else:
        raise ValueError("Unknown functionID")

    counters["f"] += 1
    return fval


def getGradient(x, functionID):
    counters["g"] += 1
    A = np.zeros((2, 2))
    b = np.zeros((2, 1))
    if functionID == 1:
        A = np.array([[8, 5], [5, 6]])
        b = np.array([4, -3])
        return A @ x + b
    elif functionID == 2:
        g1 = x[0] + 2 * x[1] * np.sin(x[0]) - 0.5 * x[1]
        g2 = x[1] - 2 * np.cos(x[0]) - 10 * np.cos(x[1]) - 0.5 * x[0]
        return np.array([g1, g2])
    elif functionID == 3:
        g1 = 0.2 * x[0] + 0.3 * x[1] * np.sin(x[0]) - 0.1 * x[1]
        g2 = 0.2 * x[1] - 0.3 * np.cos(x[0]) - 3 * np.cos(x[1]) - 0.1 * x[0]
        return np.array([g1, g2])


def getHessian(x, functionID):
    counters["H"] += 1
    if functionID == 1:
        A = np.array([[8, 5], [5, 6]])
        return A
    elif functionID == 2:
        h11 = 1 + 2 * x[1] * np.cos(x[0])
        h22 = 1 + 10 * np.sin(x[1])
        h12 = 2 * np.sin(x[0]) - 1/2
        h21 = h12
        H = np.array([[h11, h12], [h21, h22]])
        return H
    elif functionID == 3:
        h11 = 0.2 + 0.3 * x[1] * np.cos(x[0])
        h22 = 0.2 + 3 * np.sin(x[1])
        h12 = 0.3 * np.sin(x[0]) - 0.1
        h21 = h12
        H = np.array([[h11, h12], [h21, h22]])
        return H
