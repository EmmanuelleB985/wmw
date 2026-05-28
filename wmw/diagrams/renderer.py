from __future__ import annotations
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Rectangle, Circle, Polygon
import numpy as np


_DPI = 150
_FIGSIZE = (6, 4.5)
_BG = "#FAFAFA"
_OBJECT_COLOR = "#4A90D9"
_FORCE_COLOR = "#D94A4A"
_VELOCITY_COLOR = "#2ECC71"
_LABEL_SIZE = 11
_TITLE_SIZE = 12
_ARROW_STYLE = "Simple,tail_width=1.5,head_width=8,head_length=6"
_FORCE_STYLE = "Simple,tail_width=2,head_width=10,head_length=8"


def _setup_ax(ax, title=""):
    ax.set_aspect("equal")
    ax.set_facecolor(_BG)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE, fontweight="bold", pad=10)


def _force_arrow(ax, start, end, label, color=_FORCE_COLOR, offset=(5, 5)):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    mid = ((start[0]+end[0])/2 + offset[0]/50,
           (start[1]+end[1])/2 + offset[1]/50)
    ax.text(mid[0], mid[1], label, fontsize=_LABEL_SIZE-1, color=color,
            fontweight="bold", ha="center")


def _draw_ground(ax, x_range, y=0, hatch_depth=0.15):
    ax.plot(x_range, [y, y], "k-", lw=2)
    for x in np.arange(x_range[0], x_range[1], 0.15):
        ax.plot([x, x - 0.1], [y, y - hatch_depth], "k-", lw=0.8)


def _render_inclined_plane(params: dict, ax):
    theta = params.get("incline_angle_deg", 30)
    mass = params.get("mass_kg", 5)
    mu = params.get("friction_coeff", 0)
    theta_rad = math.radians(theta)


    L = 3.0
    H = L * math.sin(theta_rad)
    W = L * math.cos(theta_rad)
    incline = Polygon([[0, 0], [W, 0], [0, H]], closed=True,
                       facecolor="#E8E8E8", edgecolor="black", lw=2)
    ax.add_patch(incline)
    _draw_ground(ax, [-0.3, W + 0.3])


    arc = Arc((0, 0), 0.8, 0.8, angle=0, theta1=0, theta2=theta,
              color="black", lw=1.5)
    ax.add_patch(arc)
    ax.text(0.5, 0.15, f"{theta}°", fontsize=_LABEL_SIZE-1)


    bx = W * 0.45
    by = H * 0.45
    block_size = 0.4

    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    corners = []
    offsets = [(0, 0), (block_size, 0), (block_size, block_size), (0, block_size)]
    for dx, dy in offsets:
        rx = bx + dx * cos_t - dy * sin_t
        ry = by - dx * sin_t + dy * cos_t

        rx = bx + dx * cos_t + dy * sin_t * (-1 if theta < 90 else 1)
        ry = by + dx * sin_t * (-1 if theta > 90 else 1) + dy * cos_t
        corners.append([rx, ry])

    cx = W * 0.5
    cy = H * 0.5
    block = plt.Rectangle((cx - block_size/2, cy - block_size/2 + 0.05),
                           block_size, block_size, angle=-theta,
                           rotation_point="center",
                           facecolor=_OBJECT_COLOR, edgecolor="black", lw=1.5, zorder=3)
    ax.add_patch(block)
    ax.text(cx, cy + 0.02, f"{mass} kg", fontsize=_LABEL_SIZE-1,
            ha="center", va="center", color="white", fontweight="bold", zorder=4)


    g_len = 0.8
    _force_arrow(ax, (cx, cy), (cx, cy - g_len), f"mg = {mass*9.8:.0f} N",
                 offset=(15, 0))

    nx = cx - 0.6 * sin_t
    ny = cy + 0.6 * cos_t
    _force_arrow(ax, (cx, cy), (nx, ny), "N", color="#8B4513", offset=(-10, 5))

    if mu > 0:
        fx = cx + 0.5 * cos_t
        fy = cy + 0.5 * sin_t
        _force_arrow(ax, (cx, cy), (fx, fy), f"f (μ={mu})", color="#FF8C00",
                     offset=(5, 10))

    ax.set_xlim(-0.5, W + 0.8)
    ax.set_ylim(-0.5, H + 1.0)
    _setup_ax(ax, f"Inclined Plane ({theta}°, {mass} kg)")


def _render_projectile(params: dict, ax):
    v0 = params.get("v0_ms", 20)
    theta = params.get("angle_deg", 45)
    g = 9.8
    theta_rad = math.radians(theta)


    t_total = 2 * v0 * math.sin(theta_rad) / g
    t = np.linspace(0, t_total, 100)
    x = v0 * math.cos(theta_rad) * t
    y = v0 * math.sin(theta_rad) * t - 0.5 * g * t**2

    ax.plot(x, y, "b-", lw=2, zorder=2)
    ax.fill_between(x, 0, y, alpha=0.05, color="blue")
    _draw_ground(ax, [-0.5, max(x) * 1.1])


    vx0 = v0 * math.cos(theta_rad) * 0.04
    vy0 = v0 * math.sin(theta_rad) * 0.04
    _force_arrow(ax, (0, 0), (vx0, vy0),
                 f"v₀ = {v0} m/s", color=_VELOCITY_COLOR, offset=(5, 8))


    arc = Arc((0, 0), max(x)*0.15, max(x)*0.15, angle=0,
              theta1=0, theta2=theta, color="gray", lw=1.5)
    ax.add_patch(arc)
    ax.text(max(x)*0.08, max(x)*0.03, f"{theta}°", fontsize=_LABEL_SIZE-1)


    peak_idx = np.argmax(y)
    ax.plot(x[peak_idx], y[peak_idx], "ro", markersize=6, zorder=3)
    ax.annotate(f"h_max = {y[peak_idx]:.1f} m", xy=(x[peak_idx], y[peak_idx]),
                xytext=(x[peak_idx] + max(x)*0.05, y[peak_idx] + max(y)*0.1),
                fontsize=_LABEL_SIZE-2, arrowprops=dict(arrowstyle="->", color="gray"))


    ax.annotate("", xy=(max(x), -max(y)*0.05), xytext=(0, -max(y)*0.05),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(max(x)/2, -max(y)*0.12, f"R = {max(x):.1f} m",
            fontsize=_LABEL_SIZE-1, ha="center", color="gray")

    ax.set_xlim(-max(x)*0.1, max(x)*1.15)
    ax.set_ylim(-max(y)*0.25, max(y)*1.3)
    _setup_ax(ax, f"Projectile Motion (v₀={v0} m/s, θ={theta}°)")


def _render_collision(params: dict, ax):
    m1 = params.get("m1_kg", 5)
    m2 = params.get("m2_kg", 3)
    v1 = params.get("v1_ms", 10)
    v2 = params.get("v2_ms", 0)


    r1, r2 = 0.4, 0.35
    x1, x2 = 1.5, 3.5
    c1 = Circle((x1, 0.5), r1, facecolor=_OBJECT_COLOR, edgecolor="black", lw=2, zorder=3)
    c2 = Circle((x2, 0.5), r2, facecolor="#E67E22", edgecolor="black", lw=2, zorder=3)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(x1, 0.5, f"{m1}\nkg", ha="center", va="center",
            fontsize=_LABEL_SIZE-2, color="white", fontweight="bold", zorder=4)
    ax.text(x2, 0.5, f"{m2}\nkg", ha="center", va="center",
            fontsize=_LABEL_SIZE-2, color="white", fontweight="bold", zorder=4)


    if v1 != 0:
        _force_arrow(ax, (x1, 1.1), (x1 + 0.7, 1.1),
                     f"v₁ = {v1} m/s", color=_VELOCITY_COLOR, offset=(0, 8))
    if v2 != 0:
        _force_arrow(ax, (x2, 1.1), (x2 - 0.7, 1.1),
                     f"v₂ = {v2} m/s", color=_VELOCITY_COLOR, offset=(0, 8))
    else:
        ax.text(x2, 1.15, "at rest", fontsize=_LABEL_SIZE-2, ha="center", color="gray")

    _draw_ground(ax, [0.5, 4.5])
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.4, 2.0)
    _setup_ax(ax, f"Collision: {m1} kg → {m2} kg")


def _render_spring(params: dict, ax):
    k = params.get("k_Nm", 100)
    mass = params.get("mass_kg", 2)
    x_disp = params.get("x_m", 0.1)


    ax.plot([0, 0], [0, 1.5], "k-", lw=3)
    for y in np.arange(0, 1.5, 0.15):
        ax.plot([0, -0.12], [y, y + 0.08], "k-", lw=0.8)


    n_coils = 8
    spring_x = np.linspace(0.15, 2.0, n_coils * 20)
    spring_y = 0.75 + 0.12 * np.sin(np.linspace(0, n_coils * 2 * np.pi, len(spring_x)))
    ax.plot(spring_x, spring_y, "k-", lw=1.5)


    block = Rectangle((2.0, 0.5), 0.6, 0.5, facecolor=_OBJECT_COLOR,
                       edgecolor="black", lw=2, zorder=3)
    ax.add_patch(block)
    ax.text(2.3, 0.75, f"{mass} kg", fontsize=_LABEL_SIZE-1, ha="center",
            va="center", color="white", fontweight="bold", zorder=4)


    _force_arrow(ax, (2.6, 0.35), (2.6 - 0.7, 0.35),
                 f"F = kx", color=_FORCE_COLOR, offset=(0, -10))


    ax.annotate("", xy=(2.3 + abs(x_disp)*3, 1.15), xytext=(2.3, 1.15),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(2.3 + abs(x_disp)*1.5, 1.25, f"x = {x_disp} m",
            fontsize=_LABEL_SIZE-2, ha="center", color="gray")

    ax.text(1.0, 0.3, f"k = {k} N/m", fontsize=_LABEL_SIZE-2, ha="center", color="gray")
    _draw_ground(ax, [-0.3, 3.5], y=0)
    ax.set_xlim(-0.4, 3.5)
    ax.set_ylim(-0.4, 1.8)
    _setup_ax(ax, f"Spring-Mass (k={k} N/m, m={mass} kg)")


def _render_pendulum(params: dict, ax):
    L = params.get("L_m", 1.0)
    theta = params.get("theta_deg", 15)
    mass = params.get("mass_kg", 1.0)
    theta_rad = math.radians(theta)


    px, py = 2, 3
    ax.plot(px, py, "ks", markersize=10, zorder=3)


    bx = px + L * 1.5 * math.sin(theta_rad)
    by = py - L * 1.5 * math.cos(theta_rad)
    ax.plot([px, bx], [py, by], "k-", lw=1.5)


    bob = Circle((bx, by), 0.2, facecolor=_OBJECT_COLOR, edgecolor="black", lw=2, zorder=3)
    ax.add_patch(bob)
    ax.text(bx, by, f"{mass}", fontsize=_LABEL_SIZE-2, ha="center", va="center",
            color="white", fontweight="bold", zorder=4)


    ax.plot([px, px], [py, py - L * 1.8], "k--", lw=0.8, alpha=0.5)


    arc = Arc((px, py), 1.0, 1.0, angle=-90, theta1=0, theta2=theta,
              color="gray", lw=1.5)
    ax.add_patch(arc)
    ax.text(px + 0.15, py - 0.7, f"θ = {theta}°", fontsize=_LABEL_SIZE-1)


    _force_arrow(ax, (bx, by), (bx, by - 0.6), "mg", offset=(10, 0))

    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.5, 3.5)
    _setup_ax(ax, f"Pendulum (L={L} m, θ={theta}°)")


def _render_circuit(params: dict, ax):
    R_total = params.get("R_total_ohm",
                         params.get("R1_ohm", 10) + params.get("R2_ohm", 20))
    V = params.get("V_battery", 12)


    ax.plot([0.5, 0.5], [0.5, 2.5], "k-", lw=2)
    ax.plot([0.5, 1.0], [2.5, 2.5], "k-", lw=2)

    ax.plot([1.0, 1.0], [2.3, 2.7], "k-", lw=3)
    ax.plot([1.2, 1.2], [2.35, 2.65], "k-", lw=1.5)
    ax.text(1.1, 2.85, f"V = {V} V", fontsize=_LABEL_SIZE-1, ha="center")


    ax.plot([1.2, 3.5], [2.5, 2.5], "k-", lw=2)


    rx = np.array([3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5])
    ry = np.array([2.5, 2.7, 2.3, 2.7, 2.3, 2.7, 2.3, 2.7, 2.3, 2.7, 2.5])
    ax.plot(rx, ry, "k-", lw=2)
    ax.text(4.0, 2.9, f"R = {R_total} Ω", fontsize=_LABEL_SIZE-1, ha="center")


    ax.plot([4.5, 4.5], [2.5, 0.5], "k-", lw=2)

    ax.plot([0.5, 4.5], [0.5, 0.5], "k-", lw=2)


    ax.annotate("", xy=(2.5, 2.65), xytext=(1.8, 2.65),
                arrowprops=dict(arrowstyle="-|>", color=_VELOCITY_COLOR, lw=2))
    I = V / R_total if R_total > 0 else 0
    ax.text(2.15, 2.8, f"I = {I:.2f} A", fontsize=_LABEL_SIZE-1,
            color=_VELOCITY_COLOR, ha="center")

    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 3.5)
    _setup_ax(ax, f"Circuit (V={V} V, R={R_total} Ω)")


def _render_free_fall(params: dict, ax):
    h = params.get("h_m", 10)
    mass = params.get("mass_kg", 2)


    ax.annotate("", xy=(0.5, 0), xytext=(0.5, 2.5),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(0.3, 1.25, f"h = {h} m", fontsize=_LABEL_SIZE-1, ha="center",
            rotation=90, color="gray")


    obj = Circle((2, 2.5), 0.25, facecolor=_OBJECT_COLOR, edgecolor="black", lw=2, zorder=3)
    ax.add_patch(obj)
    ax.text(2, 2.5, f"{mass}", fontsize=_LABEL_SIZE-1, ha="center", va="center",
            color="white", fontweight="bold", zorder=4)


    _force_arrow(ax, (2, 2.2), (2, 1.4), f"g = 9.8 m/s²", offset=(15, 0))


    ax.plot([2, 2], [2.2, 0.1], "b--", lw=1, alpha=0.4)


    ghost = Circle((2, 0.25), 0.25, facecolor=_OBJECT_COLOR, edgecolor="black",
                    lw=1, alpha=0.3, zorder=2)
    ax.add_patch(ghost)

    _draw_ground(ax, [0, 3.5])
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(-0.5, 3.2)
    _setup_ax(ax, f"Free Fall (m={mass} kg, h={h} m)")


def _render_buoyancy(params: dict, ax):
    rho_fluid = params.get("rho_fluid", 1000)
    mass = params.get("mass_kg", 5)
    V_obj = params.get("V_m3", 0.005)


    water = Rectangle((0, 0), 4, 2, facecolor="#AED6F1", edgecolor="black",
                        lw=1.5, alpha=0.5)
    ax.add_patch(water)
    ax.text(0.3, 0.2, f"ρ = {rho_fluid} kg/m³", fontsize=_LABEL_SIZE-2, color="#2471A3")


    obj = Rectangle((1.5, 0.6), 1.0, 0.8, facecolor=_OBJECT_COLOR,
                      edgecolor="black", lw=2, zorder=3)
    ax.add_patch(obj)
    ax.text(2.0, 1.0, f"{mass} kg", fontsize=_LABEL_SIZE-1, ha="center",
            va="center", color="white", fontweight="bold", zorder=4)


    _force_arrow(ax, (2.0, 0.6), (2.0, -0.2), f"mg", offset=(12, 0))
    _force_arrow(ax, (2.0, 1.4), (2.0, 2.2), f"F_b", color="#2ECC71", offset=(-12, 0))


    wave_x = np.linspace(0, 4, 50)
    wave_y = 2 + 0.05 * np.sin(wave_x * 8)
    ax.plot(wave_x, wave_y, "b-", lw=1.5)

    ax.set_xlim(-0.3, 4.5)
    ax.set_ylim(-0.6, 2.8)
    _setup_ax(ax, f"Buoyancy (m={mass} kg)")


def _render_wave(params: dict, ax):
    f = params.get("f_Hz", 5)
    wavelength = params.get("wavelength_m", 0.5)
    A = params.get("A_m", 0.1)

    x = np.linspace(0, 3 * wavelength, 300)
    y = A * 10 * np.sin(2 * np.pi * x / wavelength)

    ax.plot(x, y, "b-", lw=2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")


    ax.annotate("", xy=(wavelength, A*10*1.2), xytext=(0, A*10*1.2),
                arrowprops=dict(arrowstyle="<->", color=_FORCE_COLOR, lw=1.5))
    ax.text(wavelength/2, A*10*1.4, f"λ = {wavelength} m",
            fontsize=_LABEL_SIZE-1, ha="center", color=_FORCE_COLOR)


    ax.annotate("", xy=(wavelength*0.25, A*10), xytext=(wavelength*0.25, 0),
                arrowprops=dict(arrowstyle="<->", color=_VELOCITY_COLOR, lw=1.5))
    ax.text(wavelength*0.25 - wavelength*0.15, A*5, f"A",
            fontsize=_LABEL_SIZE-1, color=_VELOCITY_COLOR)

    ax.text(max(x)*0.7, -A*10*1.5, f"f = {f} Hz, v = {f*wavelength:.1f} m/s",
            fontsize=_LABEL_SIZE-2, color="gray")

    ax.set_xlim(-wavelength*0.1, max(x)*1.05)
    ax.set_ylim(-A*10*2, A*10*2)
    _setup_ax(ax, f"Wave (f={f} Hz, λ={wavelength} m)")


def _render_optics(params: dict, ax):
    f_lens = params.get("f_m", 0.2)
    d_obj = params.get("d_obj_m", 0.4)


    lens_x = 3.0
    ax.plot([lens_x, lens_x], [-1.5, 1.5], "b-", lw=2)

    ax.plot([lens_x-0.1, lens_x, lens_x+0.1], [1.3, 1.5, 1.3], "b-", lw=2)
    ax.plot([lens_x-0.1, lens_x, lens_x+0.1], [-1.3, -1.5, -1.3], "b-", lw=2)


    ax.axhline(0, color="gray", lw=0.5, ls="--")


    ax.plot(lens_x - f_lens*5, 0, "r+", markersize=10, mew=2)
    ax.plot(lens_x + f_lens*5, 0, "r+", markersize=10, mew=2)
    ax.text(lens_x - f_lens*5, -0.3, "F", fontsize=_LABEL_SIZE-1, ha="center", color="red")
    ax.text(lens_x + f_lens*5, -0.3, "F'", fontsize=_LABEL_SIZE-1, ha="center", color="red")


    obj_x = lens_x - d_obj * 5
    ax.annotate("", xy=(obj_x, 1.0), xytext=(obj_x, 0),
                arrowprops=dict(arrowstyle="-|>", color=_OBJECT_COLOR, lw=2.5))
    ax.text(obj_x - 0.2, 0.5, "Object", fontsize=_LABEL_SIZE-2, rotation=90,
            ha="center", color=_OBJECT_COLOR)


    if d_obj != f_lens:
        d_img = 1 / (1/f_lens - 1/d_obj) if abs(1/f_lens - 1/d_obj) > 0.001 else 10
        img_x = lens_x + d_img * 5
        magnification = -d_img / d_obj
        img_h = magnification * 1.0
        if 0 < img_x < 7:
            ax.annotate("", xy=(img_x, img_h), xytext=(img_x, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#E67E22", lw=2.5))
            ax.text(img_x + 0.2, img_h/2, "Image", fontsize=_LABEL_SIZE-2,
                    rotation=90, ha="center", color="#E67E22")

    ax.text(lens_x, 1.7, f"f = {f_lens} m", fontsize=_LABEL_SIZE-1, ha="center")
    ax.set_xlim(0, 6)
    ax.set_ylim(-2, 2.2)
    _setup_ax(ax, f"Thin Lens (f={f_lens} m, d={d_obj} m)")


def _render_lever(params: dict, ax):
    F_effort = params.get("F_effort_N", 50)
    d_effort = params.get("d_effort_m", 2)
    d_load = params.get("d_load_m", 1)


    total = d_effort + d_load
    scale = 4.0 / total
    ax.plot([0.5, 0.5 + total * scale], [1.5, 1.5], "k-", lw=3)


    fx = 0.5 + d_effort * scale
    fulcrum = Polygon([[fx - 0.2, 1.5], [fx + 0.2, 1.5], [fx, 1.1]],
                       closed=True, facecolor="#95A5A6", edgecolor="black", lw=1.5)
    ax.add_patch(fulcrum)
    ax.text(fx, 0.95, "▲", fontsize=8, ha="center")


    _force_arrow(ax, (0.7, 2.2), (0.7, 1.6), f"F = {F_effort} N",
                 color=_VELOCITY_COLOR, offset=(-15, 0))

    F_load = F_effort * d_effort / d_load if d_load > 0 else 0
    load_x = 0.5 + total * scale - 0.2
    _force_arrow(ax, (load_x, 1.5), (load_x, 0.9),
                 f"W = {F_load:.0f} N", offset=(12, 0))


    ax.annotate("", xy=(0.5, 1.3), xytext=(fx, 1.3),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1))
    ax.text((0.5 + fx)/2, 1.15, f"{d_effort} m", fontsize=_LABEL_SIZE-2,
            ha="center", color="gray")

    _draw_ground(ax, [0, 5.5], y=0.9)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0.5, 2.8)
    _setup_ax(ax, "Lever")


def _render_generic(params: dict, ax, family: str):
    ax.text(0.5, 0.5, f"Physics scenario:\n{family}",
            transform=ax.transAxes, fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8E8E8"))


    y = 0.35
    for k, v in list(params.items())[:6]:
        ax.text(0.5, y, f"{k} = {v}", transform=ax.transAxes,
                fontsize=_LABEL_SIZE-1, ha="center", va="center")
        y -= 0.06

    _setup_ax(ax, family.replace("_", " ").title())


_RENDERERS = {
    "inclined_plane": _render_inclined_plane,
    "projectile": _render_projectile,
    "collision": _render_collision,
    "spring": _render_spring,
    "pendulum": _render_pendulum,
    "circuit": _render_circuit,
    "free_fall": _render_free_fall,
    "buoyancy": _render_buoyancy,
    "wave": _render_wave,
    "optics": _render_optics,
    "lever": _render_lever,
}


def render_diagram(
    family: str,
    params: dict,
    output_path: str | Path | None = None,
    dpi: int = _DPI,
) -> Path | None:
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE)
    fig.patch.set_facecolor(_BG)

    renderer = _RENDERERS.get(family)
    if renderer:
        try:
            renderer(params, ax)
        except Exception as e:
            _render_generic(params, ax, family)
    else:
        _render_generic(params, ax, family)

    if output_path is None:
        out_dir = Path("data/diagrams")
        out_dir.mkdir(parents=True, exist_ok=True)
        import hashlib
        h = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]
        output_path = out_dir / f"{family}_{h}.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_trace_diagram(trace_dict: dict, output_dir: str | Path = "data/diagrams") -> Path | None:
    family = trace_dict.get("scenario_family", "unknown")
    params = trace_dict.get("state_0", {}).get("variables", {})


    objects = trace_dict.get("state_0", {}).get("objects", [])
    for obj in objects:
        attrs = obj.get("attributes", {})
        for k, v in attrs.items():
            if k not in params:
                params[k] = v

        if "mass" in attrs and "mass_kg" not in params:
            params["mass_kg"] = attrs["mass"]

    trace_id = trace_dict.get("id", "unknown")
    out_path = Path(output_dir) / f"{trace_id}.png"
    return render_diagram(family, params, out_path)
