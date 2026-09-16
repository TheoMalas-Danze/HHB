import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import Patch



# DPI and figure size in inches
dpi = 300
width_in = 3.54
height_in = width_in / 1.6
figsize = (width_in, height_in)  # figsize in inches
figsize_car = (width_in, width_in)

# Custom palette as RGB tuples (0-255)
Perso_palette = [
    (0, 0, 0),
    (47, 72, 88),
    (51, 101, 138),
    (134, 187, 216),
    (160, 40, 41),
    (151, 161, 105)
]

# Convert RGB 0-255 to normalized floats 0-1 for matplotlib
def rgb_to_float(rgb):
    return tuple(c / 255 for c in rgb)

Perso_palette_float = [rgb_to_float(c) for c in Perso_palette]

ORANGE = (255, 165, 0)

def lininterp(c1, c2, f, alpha=1.0):
    out = tuple(np.clip((1 - f) * np.array(c1) + f * np.array(c2), 0, 255) / 255)
    return (*out, alpha)

def colorMap(f):
    if 0 <= f < 0.5:
        return lininterp(Perso_palette[0], Perso_palette[3], 2 * f)
    elif 0.5 <= f < 0.75:
        return lininterp(Perso_palette[3], ORANGE, 4 * (f - 0.5))
    elif 0.75 <= f <= 1:
        return lininterp(ORANGE, Perso_palette[4], 4 * (f - 0.75))
    else:
        return (0, 0, 0, 1)  # fallback black

# Create a colormap by sampling colorMap
n_colors = 1000
colors = [colorMap(f) for f in np.linspace(0, 1, n_colors)]
Perso_colormap = LinearSegmentedColormap.from_list("Perso_colormap", colors)

# Linestyles mapping
linestyles = {
    'solid': '-',
    'dash': '--',
    'dot': ':',
    'dashdot': '-.'
}

# Set the new default colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=Perso_palette_float)

#plt.rcParams['image.cmap'] = Perso_colormap

plt.rcParams.update({
    'text.usetex': False,
    #####Size######
    'figure.figsize': figsize,
    'figure.dpi': dpi,
    #####Font######
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    #'font.serif' : 'Computer Modern Roman',
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    #####Grid######
    'grid.color': 'gray',
    'grid.alpha': 0.4,
    'grid.linewidth': 1,
    'axes.grid': True,
    #####Line parameters######
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'lines.markeredgewidth': 0.5,  # similar to markerstrokecolor=auto (default)
})
