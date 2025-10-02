from math import sin, cos


def getObjFVal(x, functionID):
    if functionID == 1:
        fval = 4*x[0]**2 + 5*x[0]*x[1] + 3*x[1]**2 + 4*x[0] - 3*x[1] + 5
    elif functionID == 2:
        fval = 0.5*(x[0]**2 + x[1]**2) - 2 * cos(x[0]) * \
            x[1] - 10 * sin(x[1]) - 0.5 * x[0]*x[1]

    else:
        raise ValueError("Unknown functionID")

    return fval
