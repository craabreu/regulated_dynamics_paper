import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import figstyle
import itertools

fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.0))

methods = ['regulated', 'twice-regulated', 'sinr']
ns = [1, 2, 4]
timesteps = [1, 3, 6, 9, 15, 30, 45, 90]

style = {}
style[1] = '-'
style[2] = '--'
style[4] = ':'

marker = {}
marker['sinr'] = 'o'
marker['regulated'] = 's'
marker['twice-regulated'] = '^'

color = {}
color['sinr'] = 'xkcd:green'
color['regulated'] = 'xkcd:blue'
color['twice-regulated'] = 'xkcd:orange'

label = {}
label['sinr'] = 'SIN(R)'
label['regulated'] = 'Semi-Regulated NHL'
label['twice-regulated'] = 'Regulated NHL'

xmin = 0.9
xmax = 100
ax.hlines(0, xmin, xmax)
for n, method in itertools.product(ns, methods):
    data = pd.read_csv(f'comparison_alpha_1/{method}-n{n}.csv')
    p = ax.plot(data.dt, (data.E + 42.79)/42.79,
        marker=marker[method],
        label=f'{method} (n={n})',
        linestyle=style[n],
        color=color[method],
        markeredgewidth=0.5,
        markerfacecolor='white',
    )

ax.set_xlabel('Outer time step (fs)')
ax.set_xscale('log')
ax.set_xlim(xmin, xmax)

ax.set_ylabel('Relative Deviation in the\nMean Potential Energy')
ax.set_ylim(bottom=-0.015, top=0.035)

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles,
    ['']*6 + [label[method] for method in methods],
    ncol=3,
    columnspacing=0,
    labelspacing=0.2,
    title='~~~'.join([f'n={n}' for n in ns])
)
leg._legend_box.align = "left"

plt.savefig('comparison_alpha_1')
plt.show()
