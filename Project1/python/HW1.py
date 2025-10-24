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


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from getObjFVal import getObjFVal, getGradient, getHessian
from lineSearch import getAlpha
from plotOptimizationPath import plotOptimizationPath, plotOptiValues
from counters import counters


### Parameters ###


functionID = 1
if functionID == 1:
    xinit = np.array([10.5, -5.5])  # initial point
elif functionID == 3:
    xinit = np.array([13.5, 1])  # initial point
MaxIter = 100  # Maximum number of iterations
Epsilon_grad = 1e-3  # Tolerance for the gradient norm
Epsilon_step = 1e-3  # Tolerance for the step size

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
        if np.linalg.norm(x[:, i] - x[:, i-1]) < Epsilon_step or np.linalg.norm(gradient) < Epsilon_grad:
            break

        gradient = getGradient(x[:, i], functionID)
        hessian = getHessian(x[:, i], functionID)
        s = -gradient / np.linalg.norm(gradient)  # Steepest descent direction
        alpha = getAlpha(x[:, i], s, hessian, gradient, functionID)
        x[:, i + 1] = x[:, i] + alpha * s

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 2:

    print("You chose the conjugate gradients method with Fletcher-Reeves update rule.")
    gradient_k = getGradient(x[:, 0], functionID)
    d_k = -gradient_k
    for i in range(MaxIter):
        if np.linalg.norm(x[:, i] - x[:, i-1]) < Epsilon_step or np.linalg.norm(d_k) < Epsilon_grad:
            break

        alphak = getAlpha(x[:, i], d_k, getHessian(
            x[:, i], functionID), getGradient(x[:, i], functionID), functionID)

        x_kPlus1 = x[:, i] + alphak*d_k
        gradient_kPlus1 = getGradient(x_kPlus1, functionID)

        beta_k = (np.linalg.norm(gradient_kPlus1)**2) / \
            (np.linalg.norm(gradient_k)**2)

        if functionID == 1:
            d_k = -gradient_kPlus1 + beta_k*d_k
        else:
            if i % (n) == 0:
                d_k = -gradient_kPlus1  # Re-initialize every n iterations, slide 110
            else:
                d_k = -gradient_kPlus1 + beta_k*d_k

        gradient_k = gradient_kPlus1
        x[:, i + 1] = x_kPlus1

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 3:

    print("You chose the BFGS Quasi-Newton method.")
    x_k = x[:, 0]
    H_k = np.eye(n)  # Identity matrix
    g_k = getGradient(x_k, functionID)
    delta_k = -(H_k @ g_k) * getAlpha(x_k, -H_k @ g_k,
                                      H_k, g_k, functionID, searchType=4)
    delta_k = delta_k
    for i in range(MaxIter):
        x_kPlus1 = (x_k + delta_k)
        x[:, i + 1] = x_kPlus1
        if np.linalg.norm(x[:, i] - x[:, i-1]) < Epsilon_step or np.linalg.norm(g_k) < Epsilon_grad:
            break
        g_kPlus1 = getGradient(x_kPlus1, functionID)
        gamma_k = (g_kPlus1 - g_k)

        H_k = H_k + (1 + (gamma_k.T @ H_k @ gamma_k)/(delta_k.T @
                                                      gamma_k)) * (np.outer(delta_k, delta_k)/(delta_k.T @ gamma_k)) - ((np.outer(delta_k, gamma_k) @ H_k + H_k @ np.outer(gamma_k, delta_k))/(delta_k.T @ gamma_k))
        # print("iteration:", i+1)
        # print(x_kPlus1)
        # print(H_k)
        x_k = x_kPlus1
        g_k = g_kPlus1
        delta_k = (-H_k @ g_k)
        delta_k = delta_k * getAlpha(x_k, -H_k @ g_k,
                                     H_k, g_k, functionID, searchType=4)

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step


print(f'Number of iterations: {x.shape[1]-1}.')
print(f'Number of function evaluations: {counters["f"]}.')
print(f'Number of gradient evaluations: {counters["g"]}.')
print(f'Number of Hessian evaluations: {counters["H"]}.')


# Plot the function with the optimization path and the results

print(f'The optimal point is: x = {x[0, -1]:.5f}, y = {x[1, -1]:.5f}.')
print(
    f'The objective function value is: {getObjFVal(x[:, -1], functionID):.5f}.')

plotPath = True
if plotPath:
    plotOptimizationPath(x, functionID)


plotVal = False
if plotVal:
    plotOptiValues(x, functionID)


# plot_function_3D(functionID)
