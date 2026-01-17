############################################################
# A one-column example figure made by matplotlib
# To better illustrate the settings in matplotlib,
# we follow PHYSICAL REVIEW LETTERS guidelines [1]
# in adjusting figures
#
# References:
# [1] PHYSICAL REVIEW LETTERS, Information for Authors,
# https://journals.aps.org/prl/authors
#
# Last edit: 2025-04-19
############################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import textwrap
from scipy.stats import norm

font_dir = "fonts"  # adjust if needed

# Add all .ttf or .TTF files
for file in os.listdir(font_dir):
    if file.lower().endswith(".ttf"):
        font_path = os.path.join(font_dir, file)
        font_manager.fontManager.addfont(font_path)


def Lorentzian(x, center, FWHM, area, offset):
    """
    Return the value of the Lorentzian function
        offset + 0.5*FWHM*area / (np.pi * ( (x-center)**2 + (0.5*FWHM)**2 )      )

                           FWHM A
        offset + ───────────────────────
                  2π ((x-c)^2+(FWHM/2)^2 )

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center : scalar
        the position of the Lorentzian peak
    FWHM : scalar
        full width of half maximum (FWHM) / linewidth of the Lorentzian peak
    area : scalar
        area under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ----------
    Null

    """
    return offset + 0.5 * FWHM * area / (
        np.pi * ((x - center) ** 2 + (0.5 * FWHM) ** 2)
    )


############################################################
# One-column example
############################################################

# data
centerFreq = 1.348450e6
freqStamp = np.linspace(-100, +100, num=50)
FWHM = 20
Lorzlin = Lorentzian(
    x=freqStamp + centerFreq, center=centerFreq, FWHM=FWHM, area=1e-4, offset=1e-8
)
PSD_noise = (
    norm.rvs(
        loc=0,
        scale=1e-1 * np.sqrt(np.amax(Lorzlin)),
        size=len(freqStamp),
        random_state=None,
    )
    ** 2
)
NMR_decayspectrum = (Lorzlin + PSD_noise) * 1e6
Axion_sensitivity = 1e-12 * 1.0 / np.sqrt(NMR_decayspectrum)

# plot style
plt.rc("font", size=10)  # font size for all figures
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Make math text match Times New Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"

# plt.style.use('seaborn-dark')  # to specify different styles
# print(plt.style.available)  # if you want to know available styles

cm = 1 / 2.56  # convert cm to inch
# fig = plt.figure(figsize=(8.5 * cm, 12 * cm), dpi=300)  # initialize a figure following APS journal requirements
fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

gs = gridspec.GridSpec(nrows=2, ncols=1)  # create grid for multiple figures

# to specify heights and widths of subfigures
# width_ratios = [1, 1]
# height_ratios = [1]
# gs = gridspec.GridSpec(nrows=1, ncols=2, \
#   width_ratios=width_ratios, height_ratios=height_ratios)  # create grid for multiple figures

# # fix the margins
# left=0.171
# bottom=0.202
# right=0.952
# top=0.983
# wspace=0.24
# hspace=0.114
# fig.subplots_adjust(left=left, top=top, right=right,
#                     bottom=bottom, wspace=wspace, hspace=hspace)
# left=
# bottom=
# right=
# top=
# wspace=
# hspace=

ax00 = fig.add_subplot(gs[0, 0])
ax10 = fig.add_subplot(gs[1, 0])  # , sharex=ax00

ax00.plot(
    freqStamp,
    NMR_decayspectrum,
    label="PSD Signal Amp",
    color="tab:blue",
    alpha=1,
    linestyle="-",
)
# ax00.scatter(freqstamp, NMR_decayspectrum, marker='x', s=30, color='tab:black', alpha=1)
ax00.errorbar(
    x=freqStamp,
    y=NMR_decayspectrum,
    yerr=np.std(PSD_noise),
    fmt="s",
    color="tab:green",
    linewidth=1,
    markersize=3,
)
# ax00.step(x=, y=, where='post', label='', alpha=1)
ax00.fill_between(
    freqStamp,
    NMR_decayspectrum,
    np.amin(NMR_decayspectrum),  #  where = ,
    color="r",
    alpha=0.2,
    zorder=6,
)
# X_Y_Spline = make_interp_spline(x, y)
# X_ = np.linspace(x.min(), x.max(), 500)
# Y_ = X_Y_Spline(X_)
ax00.set_xlabel("Frequency - $1.348\\,449\\times 10^6$ (Hz)")
ax00.set_ylabel("PSD ($10^{-6}\\Phi_0^2/ \\mathrm{Hz}$)")
ax00.set_title("Pulsed-NMR Signal Amplitude")
# ax00.set_xscale('log')
# ax00.set_yscale('log')
# ax00.set_xticks([])
# ax00.set_yticks([])
# ax00.set_xticklabels(('$a$', '$valx$', '$b$'))
# ax00.set_yticklabels(('$a$', '$valx$', '$b$'))
# plt.setp(ax00.get_xticklabels(), visible=False)
# plt.setp(ax00.get_yticklabels(), visible=False)
# ax00.set_xlim(left=, right=)
# ax00.set_ylim(bottom=0, top=)
# ax00.vlines(x=taua, ymin = 1e-5, ymax = 1e3, colors='grey', linestyles='dotted', label='')
# ax00.hlines(y=1 / ((np.pi * homog0 * 1e6) + 1 / T2), xmin = 1e2, xmax = 1e6, colors='black', linestyles='dotted', label='')
# ax00.yaxis.set_major_locator(plt.NullLocator())
# ax00.xaxis.set_major_formatter(plt.NullFormatter())
# set visibility of ticks on the axes
# for tick in ax00.xaxis.get_major_ticks():
#         tick.tick1line.set_visible(False)
#         tick.tick2line.set_visible(False)
#         tick.label1.set_visible(False)
#         tick.label2.set_visible(False)
ax00.grid()  # set gird color, linewidth and etc.
# add an arrow
# ax00.arrow(x=, y=, dx=, \
#         dy=, width=0.02, head_width=0.199,head_length=0.04, color='black', \
#             edgecolor='none',length_includes_head=True, shape='full')
ax00.text(
    x=0, y=np.amax(Lorzlin) * 0.48, s="$\\Delta\\nu$", ha="center", va="top", color="k"
)
ax00.quiver(
    [0, 0],
    [np.amax(Lorzlin) * 0.5, np.amax(Lorzlin) * 0.5],
    [0.5 * FWHM, -0.5 * FWHM],
    [0, 0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    width=0.005,
)
# quiver(X, Y, U, V)
# Draws an arrow at each point (X[i], Y[i]) with direction and magnitude given by vector (U[i], V[i])

# ax00.legend(loc='upper right', ncol=1, frameon=False)  # bbox_to_anchor=(1.0, 1.0),


# Add zoom-in inset
ax00_zoomin: plt.Axes = inset_axes(
    ax00,  # parent axis
    bbox_to_anchor=(0.0, 0.0, 0.2, 1),  # (x0, y0, width, height)
    width="50%",
    height="50%",  # the dimensions of the inset plot
    loc="upper right",  # location in the bounding box
    # position x0, y0) and size (width, height) of the bounding box
    bbox_transform=ax00.transAxes,
    borderpad=0.1,  # padding (safe distance) around the inset plot
)
# bbox_to_anchor=(0, 0, 1, 1)：
# This defines the position and size of the bounding box for the inset, relative to the bbox_transform
# (x0, y0, width, height)
# (0, 0) → lower-left corner of the anchor box
# 1 → width = 100% of the reference box
# 1 → height = 100% of the reference box
# Axes (0,0) ----------
#       |               |
#       |    (0.5,0.5)  |
#       |               |
#        ------------ (1,1)

# width='50%' and height='50%':
# These specify the dimensions of the inset plot relative to the bounding box.
# '50%' means the inset will occupy 50% of the width and 50% of the height of ax00.

# loc='upper right':
# Defines the location of the inset in relation to the bbox_to_anchor (which is the bounding box where the inset will be placed).
# 'upper right' means the inset will be positioned at the upper-right corner of the parent axis.

# bbox_transform=ax00.transAxes:
# This defines the coordinate system for bbox_to_anchor.
# ax00.transAxes means that the bounding box (bbox_to_anchor) is expressed in axes-relative coordinates:
# 0 is the left of ax00, 1 is the right.
# 0 is the bottom of ax00, 1 is the top.
# Thus, the inset will be positioned inside the axes ax00, and (0.0, 0.0) is the bottom-left of the axes while (0.5, 1) ensures the width is 50% and the height is full.

# borderpad=1:
# This parameter defines the padding around the inset plot.
# borderpad=1 means there will be a 1 unit padding between the inset and the parent plot (ax00), ensuring that the inset plot is not stuck to the edge of the parent.

# Set zoom range
x1, x2 = 10, 20
y1, y2 = np.min(NMR_decayspectrum[(freqStamp > x1) & (freqStamp < x2)]), np.max(
    NMR_decayspectrum[(freqStamp > x1) & (freqStamp < x2)]
)

ax00_zoomin.plot(freqStamp, NMR_decayspectrum)
ax00_zoomin.set_xlim(x1, x2)
ax00_zoomin.set_ylim(y1, y2)
# ax00_zoomin.set_xticks([])
# ax00_zoomin.set_yticks([])

# Draw rectangle and connecting lines
mark_inset(ax00, ax00_zoomin, loc1=2, loc2=4, fc="grey", ec="0.0")
# fc: face color of the zoom box (e.g., 'none' for transparent)
# ec: edge color
# for loc1 and loc2:
# 2     1
# ┌─────┐
# │     |
# │     |
# └─────┘
# 3     4


# set visibility of spines / frames
for pos in ["right", "top", "bottom", "left"]:
    ax00.spines[pos].set_visible(True)


ax10.plot(freqStamp, freqStamp, color="tab:orange", label="plot label", alpha=1)
ax10.scatter(freqStamp, freqStamp, color="tab:red", marker="x", s=30, alpha=1)
ax10.set_xlabel("")
ax10.set_ylabel("")

# ax01.set_xscale('log')
# ax01.set_yscale('log')
# ax01.set_xlim(-2, 2)
# ax01.set_ylim(-0.05, 1.1)
# ax01.set_xticks([])
# ax01.set_yticks([])
ax10.grid()
ax10.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
ax012 = ax10.twinx()
ax012.set_ylabel("", color="tab:red")
ax012.set_yticks([])

# hist, bin_edges = np.histogram(conv_PSD, bins=30)
# for i, count in enumerate(hist):
#     if count > 0:
#         ax21.scatter(bin_edges[i+1], count, color='goldenrod', edgecolors='darkgoldenrod', linewidths=0.8, marker='o', s=2, zorder=6)


fig.suptitle("super title", wrap=True)

# adjust the distance between tick labels and the axis spines (i.e. the edge of the plot)
for i, ax in enumerate([ax00, ax10]):
    ax.tick_params(axis="y", which="both", pad=3)  # For y-axis ticks
    ax.tick_params(axis="x", which="both", pad=3)  # For x-axis ticks

# put figure index
# put figure index
for i, ax in enumerate([ax00, ax10]):
    # xleft, xright = ax.get_xlim()
    # ybottom, ytop = ax.get_ylim()
    ax.text(
        -0.013,
        1.02,
        s="(" + chr(i + ord("a")) + ")",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="k",
    )
# ha = 'left' or 'right'
# va = 'top' or 'bottom'
# ha = 'left' or 'right'
# va = 'top' or 'bottom'

#############################################################################
# put a mark of script information on the figure
# Get the script name and path automatically
script_path = os.path.abspath(__file__)

# # Add the annotation to the figure
# plt.annotate(
#     f"Generated by: {script_path}",
#     xy=(0.02, 0.02),
#     xycoords="figure fraction",
#     fontsize=3,
#     color="gray",
# )

# --- compute approx max characters per line ---
# fig.get_size_inches() gives (width, height) in inches
fig_width_in = fig.get_size_inches()[0]

# 72 points = 1 inch. fontsize is in points.
# So each character ~0.6 * fontsize points wide (rough approximation).
approx_char_width_pt = 0.6 * 4  # 4-pt font
fig_width_pt = fig_width_in * 72  # convert inches → points

max_chars = int(fig_width_pt / approx_char_width_pt)

wrapped_text = textwrap.fill(f"Generated by: {script_path}", width=max_chars)

plt.text(
    0.02,
    0.02,
    wrapped_text,
    transform=plt.gcf().transFigure,
    fontsize=6,
    color="gray",
    wrap=True,
)

# #############################################################################

plt.tight_layout()
# plt.savefig('example figure - one-column.png', transparent=False)
plt.show()

# colors from Piet Cornelies Mondrian
# RGB
# red 212 1 0
# orange 242 141 2
# light grey 233 226 228
# mid grey 173 189 201
# black 0 0 0
# blue 20 17 93
# yellow 252 215 7
# purple 56 63 131
# dark blue 0 13 47

# dark color list
# 'Dark2'
