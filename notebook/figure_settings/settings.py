import os 
import sys

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

## matplotlib extras for fine-tuning
from matplotlib import gridspec  # ?? I don't remember
from matplotlib import rc_context as rc_context  # configuration of figure environment
from matplotlib import cm  # high-level colormap tuning
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize  # colormap creation
import matplotlib.patches as mpatches  # legend art objects
import matplotlib.lines as mlines  # legend art objects
import matplotlib.ticker as ticker  # major/minor ticks
from cycler import cycler  # color cycles

ROOT =  os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(ROOT)

plt.style.reload_library()
plt.style.use([ROOT+'/science.mplstyle', ROOT+'/ieee.mplstyle'])

LABELS = {
    "BIN": "Bradley-Terry",
    "BINL": "Bradley-Terry\nwith 1-breaking",
    "HO_BT": "Placket-Luce",
    "HOL_BT": "Placket-Luce\nwith 1-breaking"
}

COLORS = {
    "BIN": "red",
    "BINL": "orange",
    "HO_BT": "blue",
    "HOL_BT": "cyan"
}
STYLES = {
    "BIN": "-",
    "BINL": "--",
    "HO_BT": "-",
    "HOL_BT": "--"
}
MARKERS = {
    "BIN": "o",
    "BINL": "s",
    "HO_BT": "o",
    "HOL_BT": "s",
}

# Default figure styling
default_rc = {
    # Legend settings
    "legend.title_fontsize": 8,
    "legend.fontsize": 6,
    "legend.labelspacing": 0.3,  # Vertical spacing of legend entries
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.borderaxespad": 2.,
    # Figure settings
    "figure.figsize": (3.375, 3.375),
    "figure.subplot.hspace": 0.0,  # Subplot adjustments
    "figure.subplot.wspace": 0.0,
    "figure.subplot.left": 0.05,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.05,
    "figure.subplot.top": 0.95,
    "figure.labelweight": 'bold',
    "figure.labelsize": 12,
    # Axes settings
    "xtick.labelsize" : 8,  # Size of annotations of X ticks
    "ytick.labelsize" : 8,  # Size of annotations of Y ticks
    "xtick.major.width": 0.6,
    "xtick.minor.width": 0.25,
    "ytick.major.width": 0.6,
    "ytick.minor.width": 0.25,
    "xtick.major.size": 4,  # "Size" means length here
    "xtick.minor.size": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    # Marker settings
    "lines.linewidth" : 0.5,  # Plotted line widths (1 is default)
    "lines.markersize" : 0.8,  # Size of markers on plotted data
    # "lines.markeredgewidth": 0.1,
    # "lines.markeredgecolor": 'black',
    "errorbar.capsize": 1,  # Size of lil bars on error bars,
    # Font settings
    "font.size": 7,  # Size of text elements
}

