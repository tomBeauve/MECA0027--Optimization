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


def getAlpha(x, s, A, g, functionID, searchType=3):

    # searchType:
    # 1: basic formula for SCQF : OK FOR SCQF
    # 2: Newton Raphson Method : NOT ok
    # 3: Secant method : ok
    # 4: Dichotomy Method : ok
    # 5: quadratic interpolation : ok

    if searchType == 1:  # basic formula for SCQF
        alpha = -(s.T @ g) / (s.T @ A @ s)
        return alpha
    elif searchType == 2:  # Newton Raphson Method
        alpha = 0
        g_k = g
        for i in range(10):
            if abs(s.T @ g_k) < 1e-5:
                break
            g_k = getGradient(x + alpha * s, functionID)
            A_k = getHessian(x + alpha * s, functionID)
            alpha = alpha - np.dot(s, g_k) / np.dot(s, A_k @ s)
        return alpha

    elif searchType == 3:  # Secant method
        alphakMinus1 = 0
        g_kMinus1 = g
        alphak = 1e-5
        g_k = getGradient(x + alphak * s, functionID)

        for i in range(10):
            if abs(s.T @ g_k) < 1e-5:
                break
            alphak = alphak - np.dot(s, g_k) * \
                (alphak - alphakMinus1)/(np.dot(s, g_k) - np.dot(s, g_kMinus1))

            g_kMinus1 = g_k
            alphakMinus1 = alphak
            g_k = getGradient(x + alphak * s, functionID)
        return alphak

    elif searchType == 4:  # Dichotomy Method
        rho = 0.5  # bissection
        stepLength = 1
        alphaLow = 0
        # assume gradient*s in alpha = 0 is negative

        # initialization
        alphaHigh = stepLength
        g_high = getGradient(x + alphaHigh * s, functionID)
        while s.T @ g_high < 0:
            alphaLow = alphaHigh
            alphaHigh = alphaHigh + stepLength
            g_high = getGradient(x + alphaHigh * s, functionID)

        while abs(alphaHigh - alphaLow) > 1e-5 and abs(s.T @ g_high) > 1e-5:
            alpha_new = alphaHigh - rho * (alphaHigh - alphaLow)
            g_new = getGradient(x + alpha_new * s, functionID)
            if s.T @ g_new < 0:
                alphaLow = alpha_new
            else:
                alphaHigh = alpha_new
                g_high = g_new
        return (alphaHigh + alphaLow) / 2  # in the middle of the interval

    elif searchType == 5:  # quadratic interpolation
        delta = 1
        alpha1 = 0
        alpha2 = delta
        alpha3 = 2 * delta
        delta = 2 * delta
        print("hello world")

        f1 = getObjFVal(x + alpha1 * s, functionID)
        f2 = getObjFVal(x + alpha2 * s, functionID)
        f3 = getObjFVal(x + alpha3 * s, functionID)
        while f3 < f2:
            alpha1 = alpha2
            alpha2 = alpha3
            f1 = f2
            f2 = f3
            alpha3 = alpha3 + delta * 2
            f3 = getObjFVal(x + alpha3 * s, functionID)
            delta = 2 * delta

        alpha4 = alpha2 + 1/2 * (alpha3 - alpha2)
        f4 = getObjFVal(x + alpha4 * s, functionID)
        if f4 > f2:
            alpha3 = alpha4
            f3 = f4
        else:
            alpha1 = alpha2
            f1 = f2
            alpha2 = alpha4
            f2 = f4
        i = 0
        while abs(f3 - f2) > 1e-5 or abs(f1 - f2) > 1e-5 or i < 20:
            r12 = alpha1**2 - alpha2**2
            r13 = alpha1**2 - alpha3**2
            r23 = alpha2**2 - alpha3**2
            s12 = alpha1 - alpha2
            s13 = alpha1 - alpha3
            s23 = alpha2 - alpha3

            alphaOpt = 1/2 * (f1 * r23 + f2 * r13 + f3 * r12) / \
                (f1 * s23 + f2 * s13 + f3 * s12)
            fOpt = getObjFVal(x + alphaOpt * s, functionID)

            if alphaOpt > alpha2 and alphaOpt < alpha3:
                if fOpt <= f2:
                    alpha1 = alpha2
                    f1 = f2
                    alpha2 = alphaOpt
                    f2 = fOpt
                else:
                    alpha3 = alphaOpt
                    f3 = fOpt
            elif alphaOpt > alpha1 and alphaOpt < alpha2:
                if fOpt <= f2:
                    alpha3 = alpha2
                    f3 = f2
                    alpha2 = alphaOpt
                    f2 = fOpt
                else:
                    alpha1 = alphaOpt
                    f1 = fOpt
            i = i + 1
        print("bye world")

        return alpha2


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
MaxIter = 100  # Maximum number of iterations
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
        alpha = getAlpha(x[:, i], s, hessian, gradient, functionID)
        x[:, i + 1] = x[:, i] + alpha * s

    x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

elif method == 2:

    print("You chose the conjugate gradients method with Fletcher-Reeves update rule.")
    gradient_k = getGradient(x[:, 0], functionID)
    d_k = -gradient_k
    for i in range(MaxIter):
        if np.linalg.norm(gradient_k) < Epsilon:
            break

        alphak = getAlpha(x[:, i], d_k, getHessian(
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
