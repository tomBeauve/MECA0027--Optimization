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
from math import log, sqrt


### Parameters ###


functionID = 2
xinit = np.array([10, 0])  # initial point
MaxIter = 1000000  # Maximum number of iterations
Epsilon = 1e-5  # Tolerance for the stop criteria
h = 1e-7  # Step for finite difference
hx = h * np.array([1, 0])
hy = h * np.array([0, 1])


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


def getAlpha(x, s, A, g, functionID, k):

    alpha = -(s.T @ g) / (s.T @ A @ s)
    return 1/sqrt(k+2)  # Just a dummy value, replace it with your code


def getGradient(x, functionID):
    f = getObjFVal(x, functionID)
    fhx = getObjFVal(x + hx, functionID)
    fhy = getObjFVal(x + hy, functionID)

    gradientx = (fhx - f)/h
    gradienty = (fhy - f)/h
    gradient = np.array([gradientx, gradienty])
    return gradient


def getHessian(x, functionID):
    f = getObjFVal(x, functionID)
    fhx = getObjFVal(x + hx, functionID)
    fhy = getObjFVal(x + hy, functionID)
    fmhx = getObjFVal(x - hx, functionID)
    fmhy = getObjFVal(x - hy, functionID)
    fhxhy = getObjFVal(x + hx + hy, functionID)
    fmhxhy = getObjFVal(x - hx + hy, functionID)
    fhxmhy = getObjFVal(x + hx - hy, functionID)
    fmhxmhy = getObjFVal(x - hx - hy, functionID)

    hxx = (fhx - 2*f + fmhx)/(h**2)
    hyy = (fhy - 2*f + fmhy)/(h**2)
    hxy = (fhxhy - fmhxhy - fhxmhy + fmhxmhy)/(4*h**2)

    hessian = np.array([[hxx, hxy], [hxy, hyy]])
    return hessian


### Methods ###


if method == 1:

    print("You chose the steepest descent method.")

    for i in range(MaxIter-1):
        gradient = getGradient(x[:, i], functionID)
        if np.linalg.norm(gradient) < Epsilon:
            break

        gradient = getGradient(x[:, i], functionID)
        hessian = getHessian(x[:, i], functionID)
        s = -gradient / np.linalg.norm(gradient)  # Steepest descent direction
        alpha = getAlpha(x, s, hessian, gradient, functionID, i)
        x[:, i + 1] = x[:, i] + alpha * s

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 2:

    print("You chose the conjugate gradients method with Fletcher-Reeves update rule.")
    gradient_k = getGradient(x[:, 0], functionID)
    d_k = -gradient_k/np.linalg.norm(gradient_k)
    for i in range(MaxIter):
        if np.linalg.norm(gradient_k) < Epsilon:
            break

        alphak = getAlpha(x, d_k, getHessian(
            x[:, i], functionID), getGradient(x[:, i], functionID), functionID, i)

        x_kPlus1 = x[:, i] + alphak*d_k
        x[:, i + 1] = x_kPlus1

        gradient_kPlus1 = getGradient(x_kPlus1, functionID)

        beta_k = (np.linalg.norm(gradient_kPlus1)**2) / \
            (np.linalg.norm(gradient_k)**2)
        d_k = -gradient_kPlus1 + beta_k*d_k
        d_k = d_k / np.linalg.norm(d_k)
        gradient_k = gradient_kPlus1

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
# plotOptimizationPath(x, functionID)
