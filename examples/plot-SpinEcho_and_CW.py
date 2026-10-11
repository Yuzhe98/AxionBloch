# $env:PYTHONPATH = "C:\Users\zhenf\D\Yu0702\Axionbloch-paper;$env:PYTHONPATH”
# This file is for plotting the free decay of the spin system under a RF pulse.
from tkinter import font

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

from axionbloch.utils import high_contrast_extended

data_dir = "scripts-and-data/"

# echo data
echo_data = np.load(data_dir + "SpinEcho_CPMG_simu.npz")
echo_Tdelta = echo_data["Tdelta_s"]
echo_T2 = echo_data["T2_s"]
echo_signal_freq_Hz = (
    float(echo_data["signalFreqRot_Hz"]) if "signalFreqRot_Hz" in echo_data else 1.0
)
echo_drive_phase_rad = (
    float(echo_data["init_phase_rad"]) if "init_phase_rad" in echo_data else 0.0
)
echo_T2star = (echo_T2 ** (-1) + echo_Tdelta ** (-1)) ** (-1)
echo_tau_s = (
    float(echo_data["tau_s"]) if "tau_s" in echo_data else 4.0 * float(echo_Tdelta)
)
echo_num_echoes = int(echo_data["num_echoes"]) if "num_echoes" in echo_data else 2
echo_t90_s = float(echo_data["t90_s"])
echo_size = min(
    len(echo_data["timeStamp_s"]),
    echo_data["B_vec"].shape[1],
    echo_data["trjry"].shape[1],
)
echo_plot_stride = max(1, echo_size // 20000)
echo_plot_indices = np.arange(0, echo_size, echo_plot_stride)
echo_timeStamp_s = echo_data["timeStamp_s"][echo_plot_indices]
echo_B_vec = echo_data["B_vec"][0, echo_plot_indices]
echo_M = echo_data["trjry"][0, echo_plot_indices]


# CW data
CW_data = np.load(data_dir + "CW_thermally_polarized_simu.npz")
CW_Tdelta = CW_data["Tdelta_s"]
CW_T2 = CW_data["T2_s"]
CW_signal_freq_Hz = float(CW_data["signalFreqRot_Hz"])
CW_drive_phase_rad = float(CW_data["init_phase_rad"])
CW_B1_rot_T = float(CW_data["B1_rot_T"])
CW_gamma_rad_s_T = float(CW_data["gamma_rad_s_T"])
CW_T2star = CW_T2

size = len(CW_data["timeStamp_s"]) - 1
CW_plot_stride = max(1, size // 5000)
CW_timeStamp_s = CW_data["timeStamp_s"][:size:CW_plot_stride]
CW_B_vec = CW_data["B_vec"][0][:size:CW_plot_stride]
CW_M = CW_data["trjry"][0][:size:CW_plot_stride]


linewidth = 1

plt.rc("font", size=10)  # font size for all figures
plt.rcParams["font.family"] = "Times New Roman"

# Make math text match Times New Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"
cm = 1 / 2.54  # convert cm to inch
fig = plt.figure(
    figsize=(2 * 8.5 * cm, 9 * cm), dpi=300
)  # initialize a figure following APS journal requirements
# #############################################################################
# to specify heights and widths of subfigures
width_ratios = [1, 1]
height_ratios = [0.5, 1, 0.3]
gs = gridspec.GridSpec(
    nrows=3, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios
)  # create grid for multiple figures
# #############################################################################
# fix the margins
left = 0.12
bottom = 0.157
right = 0.776
top = 0.888
wspace = 0.35
hspace = 0.114
fig.subplots_adjust(
    left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
)
# #############################################################################
echo_pulse_ax = fig.add_subplot(gs[0, 0])
echo_M_ax = fig.add_subplot(gs[1, 0])
echo_residual_ax = fig.add_subplot(gs[2, 0])

CW_pulse_ax = fig.add_subplot(gs[0, 1])
CW_M_ax = fig.add_subplot(gs[1, 1])
CW_residual_ax = fig.add_subplot(gs[2, 1])
echo_pulse_ax.plot(
    echo_timeStamp_s,
    echo_B_vec[:, 0] * 1e9,
    label="$B_{x}$",
    color=high_contrast_extended[0],
    linewidth=linewidth,
)
echo_pulse_ax.plot(
    echo_timeStamp_s,
    echo_B_vec[:, 1] * 1e9,
    label="$B_{y}$",
    color=high_contrast_extended[1],
    linewidth=linewidth,
)

echo_M_ax.plot(
    echo_timeStamp_s,
    echo_M[:, 0],
    label="$M_x^\\mathrm{simu}$",
    color=high_contrast_extended[2],
    linestyle="-",
    linewidth=linewidth,
    # marker="o",
    # markersize=0.9,
    # markevery=max(1, len(echo_timeStamp_s) // 250),
)
echo_M_ax.plot(
    echo_timeStamp_s,
    echo_M[:, 1],
    label="$M_y^\\mathrm{simu}$",
    color=high_contrast_extended[3],
    linestyle="-",
    linewidth=linewidth,
    # marker="o",
    # markersize=0.9,
    # markevery=max(1, len(echo_timeStamp_s) // 250),
)
echo_M_ax.plot(
    echo_timeStamp_s,
    np.hypot(echo_M[:, 0], echo_M[:, 1]),
    label="$(M_{x}^2 + M_{y}^2)^{1/2}$",
    linestyle="-",
    color=high_contrast_extended[4],
    linewidth=linewidth,
)
# T2star_envelope = np.exp(-(echo_timeStamp_s - echo_t90_s) / echo_T2star)
# CPMG toggling-frame envelope. The static detuning phase accumulates forward
# between refocusing pulses and is reversed after each pi pulse. Use the exact
# weighted packet distribution from the simulation when available; older data
# falls back to the Lorentzian exp(-|q|/Tdelta) approximation.
echo_pulse_starts_s = echo_tau_s * (2 * np.arange(echo_num_echoes) + 1)
echo_pulse_ends_s = echo_pulse_starts_s + echo_t90_s
# Count completed refocusing pulses: the k-th echo follows k pi pulses and
# therefore carries a k*pi phase shift relative to the initial FID.
echo_pulse_count = np.sum(
    echo_timeStamp_s[:, None] >= echo_pulse_ends_s[None, :], axis=1
)
echo_q_s = np.where(
    echo_timeStamp_s < echo_tau_s,
    echo_timeStamp_s,
    np.where(
        echo_timeStamp_s < 3.0 * echo_tau_s,
        2.0 * echo_tau_s - echo_timeStamp_s,
        echo_timeStamp_s - 4.0 * echo_tau_s,
    ),
)
if "detuning_rad_s" in echo_data and "homogeneity_weights" in echo_data:
    echo_detuning_rad_s = echo_data["detuning_rad_s"]
    echo_weights = echo_data["homogeneity_weights"]
    echo_weights = echo_weights / np.sum(echo_weights)
    echo_coherence = np.zeros(echo_timeStamp_s.shape, dtype=complex)
    for detuning_rad_s, weight in zip(echo_detuning_rad_s, echo_weights):
        echo_coherence += weight * np.exp(1j * detuning_rad_s * echo_q_s)
    echo_distribution_phase = np.angle(echo_coherence)
    echo_distribution_envelope = np.abs(echo_coherence)
else:
    echo_distribution_phase = np.zeros(echo_timeStamp_s.shape)
    echo_distribution_envelope = np.exp(-np.abs(echo_q_s) / echo_Tdelta)
echo_envelope = (
    np.exp(-(echo_timeStamp_s - 1 * echo_t90_s) / echo_T2) * echo_distribution_envelope
)
# Include the finite 90-degree excitation ramp and the alternating phase
# inversion produced by successive CPMG pi pulses.
if echo_t90_s > 0:
    echo_envelope *= np.where(
        echo_timeStamp_s < echo_t90_s,
        np.sin(0.5 * np.pi * echo_timeStamp_s / echo_t90_s),
        1.0,
    )
echo_phase = (
    2 * np.pi * echo_signal_freq_Hz * (echo_timeStamp_s)
    + echo_drive_phase_rad
    + np.pi / 2
    + echo_pulse_count * np.pi
    + echo_distribution_phase
)
echo_M_theory = np.column_stack(
    (echo_envelope * np.cos(echo_phase), echo_envelope * np.sin(echo_phase))
)
# The idealized theory does not model magnetization during finite pi pulses.
for pulse_start_s, pulse_end_s in zip(echo_pulse_starts_s, echo_pulse_ends_s):
    during_pulse = (echo_timeStamp_s >= pulse_start_s) & (
        echo_timeStamp_s < pulse_end_s
    )
    echo_M_theory[during_pulse] = np.nan

echo_M_ax.plot(
    echo_timeStamp_s,
    echo_M_theory[:, 0],
    label="$M_x^\\mathrm{theo}$",
    color=high_contrast_extended[4],
    linestyle="--",
    linewidth=linewidth,
)
echo_M_ax.plot(
    echo_timeStamp_s,
    echo_M_theory[:, 1],
    label="$M_y^\\mathrm{theo}$",
    color=high_contrast_extended[5],
    linestyle="--",
    linewidth=linewidth,
)
# echo_M_ax.plot(
#     echo_timeStamp_s[: len(echo_timeStamp_s) // 4],
#     T2star_envelope[: len(echo_timeStamp_s) // 4],
#     # label="$e^{{-t/T_2^*}}$",
#     linestyle="dashed",
#     color="k",
#     linewidth=linewidth,
# )

echo_M_ax.plot(
    echo_timeStamp_s[100 : len(echo_timeStamp_s) // 5],
    echo_envelope[100 : len(echo_timeStamp_s) // 5],
    # label="$M_x^\\mathrm{theo}$",
    color="k",
    linestyle="--",
    linewidth=linewidth,
)

echo_M_ax.text(
    2.1,
    0.185,
    # "$e^{\\frac{-t}{T_2^*}}$",
    "$e^{-t/T_2^*}$",
    color="k",
    fontsize=10,
    ha="left",
)
T2_envelope = np.exp(-(echo_timeStamp_s - echo_t90_s) / echo_T2)
echo_M_ax.plot(
    echo_timeStamp_s,
    T2_envelope,
    # label="$e^{{-t/T_2}}$",
    linestyle="dashed",
    color=high_contrast_extended[8],
    linewidth=linewidth,
)
echo_M_ax.text(
    11.3,
    0.4,
    "$e^{-t/T_2}$",
    color=high_contrast_extended[8],
    fontsize=10,
    ha="left",
)

echo_residual_ax.plot(
    echo_timeStamp_s,
    echo_M[:, 0] - echo_M_theory[:, 0],
    label="$M_x^\\mathrm{simu}-M_x^\\mathrm{theo}$",
    color=high_contrast_extended[6],
    linewidth=linewidth,
)
echo_residual_ax.plot(
    echo_timeStamp_s,
    echo_M[:, 1] - echo_M_theory[:, 1],
    label="$M_y^\\mathrm{simu}-M_y^\\mathrm{theo}$",
    color=high_contrast_extended[7],
    linewidth=linewidth,
)


CW_pulse_ax.plot(
    CW_timeStamp_s,
    CW_B_vec[:, 0] * 1e9,
    label="$B_{x}$",
    color=high_contrast_extended[0],
    linewidth=linewidth,
)
CW_pulse_ax.plot(
    CW_timeStamp_s,
    CW_B_vec[:, 1] * 1e9,
    label="$B_{y}$",
    color=high_contrast_extended[1],
    linewidth=linewidth,
)

CW_M_ax.plot(
    CW_timeStamp_s,
    CW_M[:, 0],
    label="$M_x^\\mathrm{simu}$",
    color=high_contrast_extended[2],
    linewidth=linewidth,
    linestyle="-",
    # marker="o",
    # markersize=0.2,
    # markevery=max(1, len(CW_timeStamp_s) // 250),
)
CW_M_ax.plot(
    CW_timeStamp_s,
    CW_M[:, 1],
    label="$M_y^\\mathrm{simu}$",
    color=high_contrast_extended[3],
    linestyle="-",
    linewidth=linewidth,
    # marker="o",
    # markersize=0.2,
    # markevery=max(1, len(CW_timeStamp_s) // 250),
)

# Build the theoretical CW signals from the relaxation envelope and carrier.
print(f"CW_T2star = {CW_T2star:.3f} s")
print(f"CW_B1_rot_T = {CW_B1_rot_T:.3e} T")
CW_envelope = (1 - np.exp(-CW_timeStamp_s / CW_T2star)) * np.sin(
    CW_gamma_rad_s_T * CW_B1_rot_T * CW_T2star
)
CW_phase = (
    2 * np.pi * CW_signal_freq_Hz * CW_timeStamp_s + CW_drive_phase_rad + np.pi / 2
)
CW_M_theory = np.column_stack(
    (CW_envelope * np.cos(CW_phase), CW_envelope * np.sin(CW_phase))
)


# envelope
CW_M_ax.plot(
    CW_timeStamp_s,
    CW_envelope,
    # label="$\\gamma B T_2^*\\times$\n$(1-e^{-t/T_2^*})$",
    linestyle="dashed",
    color="#393b79",
    linewidth=linewidth,
)
# -----
CW_M_ax.text(
    3.5,
    0.0018,
    # "$\\gamma B T_2^* (1-e^{\\frac{-t}{T_2^*}})$",
    "$\\gamma B T_2^* (1-e^{-t/T_2^*})$",
    color="#393b79",
    ha="left",
)


CW_M_ax.plot(
    CW_timeStamp_s,
    CW_M_theory[:, 0],
    label="$M_x^\\mathrm{theo}$",
    color=high_contrast_extended[4],
    linestyle="--",
    linewidth=linewidth,
)
CW_M_ax.plot(
    CW_timeStamp_s,
    CW_M_theory[:, 1],
    label="$M_y^\\mathrm{theo}$",
    color=high_contrast_extended[5],
    linestyle="--",
    linewidth=linewidth,
)
# T2_envelope = np.exp(-(timeStamp_s + 0.02) / T2)
# M_ax.plot(
#     timeStamp_s,
#     T2_envelope,
#     label=f"$e^{{-t/T_2}}$",
#     linestyle="dashed",
#     color=high_contrast_extended[5],
#     linewidth = linewidth ,
# )
CW_residual_ax.plot(
    CW_timeStamp_s,
    CW_M[:, 0] - CW_M_theory[:, 0],
    label="$M_x^\\mathrm{simu}-M_x^\\mathrm{theo}$",
    color=high_contrast_extended[6],
    linewidth=linewidth,
)
CW_residual_ax.plot(
    CW_timeStamp_s,
    CW_M[:, 1] - CW_M_theory[:, 1],
    label="$M_y^\\mathrm{simu}-M_y^\\mathrm{theo}$",
    color=high_contrast_extended[7],
    linewidth=linewidth,
)

# Axis labels and limits are kept together here for easy figure-wide editing.
echo_xlim = None
CW_xlim = None
axis_settings = (
    (echo_pulse_ax, "", "$B\\,(\\mathrm{n T})$", echo_xlim, None),
    (echo_M_ax, "", "$M\\,(M_\\mathrm{eqb})$", echo_xlim, (-1.1, 1.1)),
    (
        echo_residual_ax,
        "Time (s)",
        "Residual $(M_\\mathrm{eqb})$",
        echo_xlim,
        (-1.2e-2, 1.2e-2),
    ),
    (CW_pulse_ax, "", "", CW_xlim, (-8e-3, 8e-3)),
    (CW_M_ax, "", "", CW_xlim, (-0.00223, 0.00255)),
    (CW_residual_ax, "Time (s)", "", CW_xlim, (-1.2e-6, 1.2e-6)),
)
for ax, xlabel, ylabel, xlim, ylim in axis_settings:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

# hide x-axis tick labels for the upper plot
echo_pulse_ax.set_xticklabels([])
echo_M_ax.set_xticklabels([])
CW_pulse_ax.set_xticklabels([])
CW_M_ax.set_xticklabels([])

echo_residual_ax.set_yticks([-1e-2, 0, 1e-2])
echo_residual_ax.set_yticklabels(["$-10^{-2}$", "$0$", "$10^{-2}$"])

CW_residual_ax.set_yticks([-1e-6, 0, 1e-6])
CW_residual_ax.set_yticklabels(["$-10^{-6}$", "$0$", "$10^{-6}$"])

# set legends for the subplots
CW_pulse_ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.0, 1.0),
    frameon=False,
)
CW_M_ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.0, 1.0),
    frameon=False,
)
CW_residual_ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.0, 1.0),
    frameon=False,
)


# put figure index
letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
for i, ax in enumerate([echo_pulse_ax, CW_pulse_ax]):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.text(-0.25, 1.2, letters[i], transform=ax.transAxes)
fig.align_ylabels([echo_pulse_ax, echo_M_ax, echo_residual_ax])
# ha = 'left' or 'right'
# va = 'top' or 'bottom'
# plt.tight_layout()
plt.savefig("tex/figures/SpinEcho_and_CW.pdf")
plt.show()
