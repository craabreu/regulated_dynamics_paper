import math
import numpy as np
import matplotlib.pyplot as plt
import figstyle
from scipy.special import gamma

cases = [1, 2, 4, 8]
colors = ['r', 'g', 'b', 'm']

def plot_H_contribution(index, x, alpha):
    ax[index].set_ylabel('$\\mathcal{H}$ contribution ($kT$)')
    for n, color in zip(cases, colors):
        ax[index].plot(x, n*np.log(np.cosh(alpha(n)*x)), f'{color}-', label=f'$n={n}$')
    ax[index].plot(x, x**2/2, 'k-', label='standard')
    ax[index].set_ylim([0, 2])
    ax[index].legend(loc='upper left')

def plot_velocity(index, x, alpha):
    for n, color in zip(cases, colors):
        c = np.sqrt(n*alpha(n))
        ax[index].plot(x, c*np.tanh(np.sqrt(alpha(n)/n)*x), f'{color}-', label=f'$n={n}$')
    ax[index].plot(x, x, 'k-', label=f'$n \\to \\infty$')
    ax[index].legend(loc='upper left')

def factor(n):
    return gamma((n+1)/2)/(gamma(n/2)*np.sqrt(math.pi))

def plot_momentum_distribution(index, x, alpha):
    def density(n):
        coeff = np.sqrt(alpha(n)/n)
        return lambda x: coeff*factor(n)/(np.cosh(coeff*x)**n)
    opacity = 0.1
    for n, color in zip(cases, colors):
        ax[index].plot(x, density(n)(x), f'{color}-', label=f'$n={n}$')
        ax[index].fill_between(x, 0, density(n)(x), color=color, alpha=opacity)
    ax[index].plot(x, np.exp(-x**2/2)/np.sqrt(2*math.pi), 'k-', label='normal')
    ax[index].fill_between(x, 0, np.exp(-x**2/2)/np.sqrt(2*math.pi), color='k', alpha=opacity)
    # ax[index].set_ylim([0, 0.42])
    ax[index].legend(loc='upper left')

def plot_velocity_distribution(index, xinp, alpha):
    def density(n):
        c = np.sqrt(alpha(n)*n)
        return lambda x: (factor(n)/c)*(1 - (x/c)**2)**(n/2-1)
    opacity = 0.1
    for n, color in zip(cases, colors):
        c = np.sqrt(alpha(n)*n)
        x = xinp[np.abs(xinp) < c]
        y = density(n)(x)
        if n < 3:
            ax[index].plot([-c, -c, np.nan, c, c], [0, y[0], np.nan, y[-1], 0], f'{color}--')
            x = np.append(np.insert(x, 0, [-c, np.nan, -c]), [c, np.nan, c])
            y = np.append(np.insert(y, 0, [0, np.nan, y[0]]), [y[-1], np.nan, 0])
        prepend = xinp[xinp < -c]
        append = xinp[xinp > c]
        x = np.append(np.insert(x, 0, prepend), append)
        y = np.append(np.insert(y, 0, np.zeros_like(prepend)), np.zeros_like(append))
        ax[index].plot(x, y, f'{color}-', label=f'$n={n}$')
        ax[index].fill_between(x, 0, y, color=color, alpha=opacity)
    ax[index].plot(xinp, np.exp(-xinp**2/2)/np.sqrt(2*math.pi), 'k-', label='normal')
    ax[index].fill_between(xinp, 0, np.exp(-xinp**2/2)/np.sqrt(2*math.pi), color='k', alpha=opacity)
    ax[index].set_ylim([0, 0.85])
    ax[index].legend(loc='upper center')


fig, axes = plt.subplots(3, 2, figsize=(7, 7), sharey='row',
                         gridspec_kw={'height_ratios': [1, 1, 1.3]})
ax = axes.flatten()
fig.subplots_adjust(hspace=0.35,wspace=0.1)

limit = 3.5
x = np.linspace(-limit, limit, 200)
for k in range(4):
    ax[k].set_xlim([-limit, limit])

plot_momentum_distribution(0, x, lambda n:1)
plot_momentum_distribution(1, x, lambda n:(n+1)/n)
# plot_momentum_distribution(1, x, lambda n:(n+0.56)/n)
plot_velocity(2, x, lambda n:1)
plot_velocity(3, x, lambda n:(n+1)/n)
# plot_velocity(3, x, lambda n:n/(n+1))

ax[2].set_ylim([-3, 3])
ax[0].set_ylabel('Probability Density')
ax[2].set_ylabel('Reduced Velocity $\\left(\\frac{v_i}{\\sqrt{kT/m_i}}\\right)$')
for k in range(4):
    ax[k].set_xlabel('Reduced Momentum $\\left(\\frac{p_i}{\\sqrt{m_i k T}}\\right)$')

limit = 2.5
x = np.linspace(-limit, limit, 400)
ax[4].set_xlim([-limit, limit])
ax[5].set_xlim([-limit, limit])

ax[4].set_xlabel('Reduced Velocity $\\left(\\frac{v_i}{\\sqrt{kT/m_i}}\\right)$')
ax[5].set_xlabel('Reduced Velocity $\\left(\\frac{v_i}{\\sqrt{kT/m_i}}\\right)$')
ax[4].set_ylabel('Probability Density')

plot_velocity_distribution(4, x, lambda n:1)
plot_velocity_distribution(5, x, lambda n:(n+1)/n)
# plot_velocity_distribution(5, x, lambda n:(n+0.56)/n)

kwargs = dict(
    xycoords='axes fraction',
    horizontalalignment='center',
    verticalalignment='center',
)
for k in range(2):
    ax[2*k].annotate('$\\alpha_n = 1$', xy=(0.5, 0.2), **kwargs)
    ax[2*k+1].annotate('$\\alpha_n = \\displaystyle{\\frac{n+1}{n}}$', xy=(0.5, 0.2), **kwargs)
ax[4].annotate('$\\alpha_n = 1$', xy=(0.5, 0.15), **kwargs)
ax[5].annotate('$\\alpha_n = \\displaystyle{\\frac{n+1}{n}}$', xy=(0.5, 0.15), **kwargs)

fig.savefig('momentum_and_velocity_functions')

plt.show()
