###########################################################################
#          MECA0027 Structural and Multidisciplinary Optimization         #
#                     Unconstrained Optimization                          #
#                    University Of Liège, Belgium                         #
###########################################################################

# Solve the minimization problem
#
#             min f(x,y)
#
# using the following optimization methods
#
# 1) Steepest descent
# 2) Conjugate gradients with Fletcher-Reeves update rule
# 3) BFGS Quasi-Newton


import numpy as np
from getObjFVal import getObjFVal
from plotOptimizationPath import plotOptimizationPath


def getAlpha(x, s, A, g, functionID):

    alpha = -(s.T @ g) / (s.T @ A @ s)
    return alpha


def getGradient(x, functionID):
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


def getHessian(x, functionID):
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

### Parameters ###


functionID = 2
xinit = np.array([1, 0])  # initial point
MaxIter = 1000  # Maximum number of iterations
Epsilon = 1e-5  # Tolerance for the stop criteria

### Initialization ###

print("Which optimization method do you want to use? Press:")
print("     1 for Steepest descent method")
print("     2 for Conjugate gradients method with Fletcher-Reeves update rule")
print("     3 for BFGS Quasi-Newton method")
method = int(input())

n = 2  # Dimension of the problem
xinit = xinit.reshape(2, 1)  # To be sure that it's a column vector
x = np.zeros((n, MaxIter+1))  # Initialization of vector x
x[:, 0] = xinit[:, 0]  # Put xinit in vector x.

### Methods ###

if method == 1:

    print("You chose the steepest descent method.")

    for i in range(MaxIter):
        gradient = getGradient(x[:, i], functionID)
        if np.linalg.norm(x[:, i] - x[:, i-1]) < Epsilon or np.linalg.norm(gradient) < Epsilon:
            break

        gradient = getGradient(x[:, i], functionID)
        hessian = getHessian(x[:, i], functionID)
        s = -gradient / np.linalg.norm(gradient)  # Steepest descent direction
        alpha = getAlpha(x, s, hessian, gradient, functionID)
        x[:, i + 1] = x[:, i] + alpha * s

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 2:

    print("You chose the conjugate gradients method with Fletcher-Reeves update rule.")
    gradient_k = getGradient(x[:, 0], functionID)
    d_k = -gradient_k
    for i in range(MaxIter):
        if np.linalg.norm(gradient_k) < Epsilon:
            break

        alphak = getAlpha(x, d_k, getHessian(
            x[:, i], functionID), getGradient(x[:, i], functionID), functionID)

        x_kPlus1 = x[:, i] + alphak*d_k
        gradient_kPlus1 = getGradient(x_kPlus1, functionID)

        beta_k = (np.linalg.norm(gradient_kPlus1)**2) / \
            (np.linalg.norm(gradient_k)**2)
        d_k = -gradient_kPlus1 + beta_k*d_k
        gradient_k = gradient_kPlus1
        x[:, i + 1] = x_kPlus1

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 3:

    print("You chose the BFGS Quasi-Newton method.")

    for i in range(MaxIter):

        # ---------------------------------------------------------------------------
        # ADD YOUR CODE
        pass  # Remove this 'pass' statement once you've added your code

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

    # Plot the function with the optimization path and the results


print(f'The optimal point is: x = {x[0, -1]:.5f}, y = {x[1, -1]:.5f}.')
print(
    f'The objective function value is: {getObjFVal(x[:, -1], functionID):.5f}.')
plotOptimizationPath(x, functionID)
