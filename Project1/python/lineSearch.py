import numpy as np
from getObjFVal import getObjFVal, getGradient, getHessian


alphaEpsilon = 1e-9


def getAlpha(x, s, A, g, functionID, searchType=3):

    # searchType:
    # 1: basic formula for SCQF : OK FOR SCQF
    # 2: Newton Raphson Method : ok for method 1 and 2 for function 1 and 2
    # 3: Secant method : ok for method 1 and 2 for function 1 and 2
    # 4: Dichotomy Method : ok for method 1 and 2 for function 1 and 2
    # 5: quadratic interpolation : ok for method 1 and 2 for function 1

    if searchType == 1:  # basic formula for SCQF
        alpha = -(s.T @ g) / (s.T @ A @ s)
        return alpha
    elif searchType == 2:  # Newton Raphson Method
        alpha = 0
        g_k = g
        for i in range(10):
            if abs(s.T @ g_k) < alphaEpsilon:
                break
            g_k = getGradient(x + alpha * s, functionID)
            A_k = getHessian(x + alpha * s, functionID)
            alpha = alpha - np.dot(s, g_k) / np.dot(s, A_k @ s)
        return alpha

    elif searchType == 3:  # Secant method
        alphakMinus1 = 0
        g_kMinus1 = g
        alphak = 3*1e-10
        g_k = getGradient(x + alphak * s, functionID)

        for i in range(100):
            if abs(alphak - alphakMinus1) < alphaEpsilon and abs(s.T @ g_k) < alphaEpsilon:
                break

            alphakPlus1 = alphak - (np.dot(s, g_k) *
                                    (alphak - alphakMinus1)/(np.dot(s, g_k) - np.dot(s, g_kMinus1)))

            g_kMinus1 = g_k
            alphakMinus1 = alphak
            alphak = alphakPlus1
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

        while abs(alphaHigh - alphaLow) > alphaEpsilon and abs(s.T @ g_high) > alphaEpsilon:
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
        while abs(f3 - f2) > alphaEpsilon or abs(f1 - f2) > alphaEpsilon or i < 20:
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

        return alpha2
