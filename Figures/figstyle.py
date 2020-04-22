import matplotlib.pyplot as plt
from cycler import cycler

params = {
	'font.size': 8,
	'text.usetex': True,
	'lines.linewidth': 1,
	'lines.markersize': 3,
	'errorbar.capsize': 2,
	'legend.frameon': False,
	'legend.columnspacing': 1,
	'savefig.format': 'png',
	'savefig.dpi': 600,
	'savefig.bbox': 'tight',
	'figure.subplot.hspace': 0.1,
	'axes.prop_cycle': cycler(
		color=['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red'],
		marker=['o', 's', '^', 'v'],
	),
}

plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-white')
plt.rcParams.update(params)
