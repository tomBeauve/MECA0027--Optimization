import numpy as np
from getObjFVal import getObjFVal, getGradient, getHessian
from lineSearch import getAlpha
from matplotlib import pyplot as plt


def opti(method, functionID, x, Epsilon_grad=1e-3, Epsilon_step=1e-6, MaxIter=100):
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


MaxIter = 100
n = 2
functionID = 1


methods = [1, 2, 3]  # 1: SD, 2: CG, 3: BFGS
method_names = ['Steepest Descent', 'Conjugate Gradient', 'BFGS']

Epsilon_grad_list = [1e-1, 1e-2, 1e-3, 1e-4,
                     1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

N_all = np.zeros((len(methods), len(Epsilon_grad_list)))

for m_idx, method in enumerate(methods):
    for i, Epsilon_grad in enumerate(Epsilon_grad_list):
        xinit = np.array([10.5, -5.5])  # initial point
        x = np.zeros((n, MaxIter + 1))
        x[:, 0] = xinit
        x_result = opti(method, functionID, x, Epsilon_grad=Epsilon_grad)
        N_all[m_idx, i] = x_result.shape[1] - 1  # number of iterations

# Plotting
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']
for m_idx, method in enumerate(methods):
    plt.scatter(Epsilon_grad_list, N_all[m_idx, :], marker=markers[m_idx],
                label=method_names[m_idx])

plt.xscale('log')
plt.xlabel(r'$\varepsilon_{\mathrm{grad}}$', fontsize=16)
plt.ylabel('N', fontsize=20)
plt.gca().invert_xaxis()  # smaller tolerances on the right
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=16, width=1.5)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.legend()
plt.tight_layout()
plt.savefig(r'plots/epsGrad_f1allMethods.pdf', dpi=300)
plt.show()
