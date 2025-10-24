import matplotlib.pyplot as plt
import numpy as np
from getObjFVal import getObjFVal, getGradient, getHessian
from lineSearch import getAlpha
from matplotlib import pyplot as plt


def opti(method, functionID, x, Epsilon_grad=1e-3, Epsilon_step=1e-3, MaxIter=100):
    n = 2
    if method == 1:

        for i in range(MaxIter):
            gradient = getGradient(x[:, i], functionID)
            if np.linalg.norm(x[:, i] - x[:, i-1]) < Epsilon_step or np.linalg.norm(gradient) < Epsilon_grad:
                break

            gradient = getGradient(x[:, i], functionID)
            hessian = getHessian(x[:, i], functionID)
            # Steepest descent direction
            s = -gradient / np.linalg.norm(gradient)
            alpha = getAlpha(x[:, i], s, hessian, gradient, functionID)
            x[:, i + 1] = x[:, i] + alpha * s

        x = x[:, :i + 1]  # Remove the zero elements due to the initialization step

    elif method == 2:

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
            x_k = x_kPlus1
            g_k = g_kPlus1
            delta_k = (-H_k @ g_k)
            delta_k = delta_k * getAlpha(x_k, -H_k @ g_k,
                                         H_k, g_k, functionID, searchType=4)

        x = x[:, :i + 1]  # Remove the zero elements due to the initialization step
    return x


n = 2
MaxIter = 100
functionID = 3


def FvalueVSIter():

    methods = [1, 2, 3]  # 1: SD, 2: CG, 3: BFGS
    method_names = ['Steepest Descent', 'Conjugate Gradient', 'BFGS']
    markers = ['o', 's', '^']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Choose initial point depending on function
    if functionID == 1:
        xinit = np.array([10.5, -5.5])
    elif functionID == 3:
        xinit = np.array([13.5, 1])

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']

    for m_idx, method in enumerate(methods):
        if functionID == 1:
            xinit = np.array([10.5, -5.5])
        elif functionID == 3:
            xinit = np.array([13.5, 1])

        x = np.zeros((n, MaxIter + 1))
        x[:, 0] = xinit
        x_result = opti(method, functionID, x)  # returns number of iterations
        N_iter = x_result.shape[1]
        f_values = [getObjFVal(x[:, k], functionID) for k in range(N_iter)]
        plt.plot(range(N_iter), f_values,
                 marker=markers[m_idx], label=method_names[m_idx])

    plt.xlabel('k', fontsize=16)
    plt.ylabel(rf'$f_{{{functionID}}}(x_k)$', fontsize=20)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/functionValue_vs_iterF' +
                f'{functionID}' + '.pdf', dpi=300)
    plt.show()


def NVSEpsilon():
    # Define the methods: 1 = Steepest Descent, 2 = CG, 3 = BFGS
    methods = [1, 2, 3]
    method_names = ['Steepest Descent', 'Conjugate Gradient', 'BFGS']
    markers = ['o', 's', '^']

    # Define the tested epsilon values (gradient norm tolerance)
    eps_values = np.logspace(-1, -9, 9)  # from 1e-1 to 1e-9 (log scale)

    # Store number of iterations for each method and epsilon
    N_all = np.zeros((len(methods), len(eps_values)))

    for m_idx, method in enumerate(methods):
        for e_idx, eps in enumerate(eps_values):

            # Set functionID and initial point
            if functionID == 1:
                xinit = np.array([10.5, -5.5])
            elif functionID == 3:
                xinit = np.array([13.5, 1])
            else:
                raise ValueError("Unsupported functionID")

            # Allocate storage for iterates
            x = np.zeros((n, MaxIter + 1))
            x[:, 0] = xinit

            # Call optimization routine with current epsilon
            x_result = opti(method, functionID, x, eps)
            N_iter = x_result.shape[1] - 1

            # Store number of iterations until convergence
            N_all[m_idx, e_idx] = N_iter

    # Plotting
    plt.figure(figsize=(9, 6))
    for m_idx, method in enumerate(methods):
        plt.scatter(
            eps_values, N_all[m_idx],
            marker=markers[m_idx],
            label=method_names[m_idx]
        )

    plt.xscale('log')
    plt.xlabel(r'$\varepsilon_{\mathrm{grad}}$', fontsize=16)
    plt.ylabel('Number of iterations $N$', fontsize=16)
    # plt.title('Convergence study with respect to $\varepsilon_{\mathrm{grad}}$', fontsize=15)
    plt.gca().invert_xaxis()  # smaller tolerances on the right
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14, width=1.2)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig('plots/epsGrad_f1allMethods.pdf', dpi=300)
    plt.show()


FvalueVSIter()
