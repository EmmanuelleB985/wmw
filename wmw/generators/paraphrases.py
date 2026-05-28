from __future__ import annotations
import random


def _inclined_plane_templates(p: dict) -> list[str]:
    m = p.get("mass_kg", 5)
    theta = p.get("incline_angle_deg", 30)
    mu = p.get("friction_coeff", 0)
    friction = f" with coefficient of friction μ = {mu}" if mu > 0 else " (frictionless)"
    return [
        f"A {m} kg block slides down a {theta}° incline{friction}. What is its acceleration?",
        f"Find the acceleration of a {m}-kg block on a {theta}° frictionless incline." if mu == 0
        else f"Find the acceleration of a {m}-kg block on a {theta}° incline (μ = {mu}).",
        f"What acceleration does a block of mass {m} kg experience on a {theta}° inclined surface{friction}?",
        f"Determine the net acceleration of a {m} kg object placed on an incline at {theta}° to the horizontal{friction}.",
        f"A {m}-kg mass is released from rest on a plane inclined at {theta}°{friction}. Calculate the acceleration.",
    ]


def _projectile_templates(p: dict) -> list[str]:
    v0 = p.get("v0_ms", 20)
    theta = p.get("angle_deg", 45)
    return [
        f"A projectile is launched at {v0} m/s at {theta}° above the horizontal. What is the range?",
        f"Find the horizontal range of a ball thrown at {v0} m/s at an angle of {theta}°.",
        f"Calculate how far a projectile travels horizontally when launched at {v0} m/s at {theta}° (no air resistance).",
        f"What is the range of a projectile with initial speed {v0} m/s and launch angle {theta}°?",
        f"An object is fired at {theta}° with a speed of {v0} m/s. How far does it land from the launch point?",
    ]


def _collision_templates(p: dict) -> list[str]:
    m1 = p.get("m1_kg", 5)
    m2 = p.get("m2_kg", 3)
    v1 = p.get("v1_ms", 10)
    v2 = p.get("v2_ms", 0)
    rest = "at rest" if v2 == 0 else f"moving at {v2} m/s"
    return [
        f"A {m1} kg object moving at {v1} m/s collides with a {m2} kg object {rest}. Find the velocity after a perfectly inelastic collision.",
        f"In a perfectly inelastic collision, a {m1}-kg mass at {v1} m/s hits a {m2}-kg mass {rest}. What is the final speed?",
        f"Two objects ({m1} kg at {v1} m/s and {m2} kg {rest}) undergo a perfectly inelastic collision. Calculate the combined velocity.",
        f"What is the final velocity when a {m1} kg ball at {v1} m/s merges with a {m2} kg ball {rest}?",
    ]


def _free_fall_templates(p: dict) -> list[str]:
    h = p.get("h_m", 10)
    m = p.get("mass_kg", 2)
    return [
        f"A {m} kg object is dropped from a height of {h} m. What is its velocity just before hitting the ground?",
        f"Find the speed of a {m}-kg mass after falling {h} m from rest (no air resistance).",
        f"Calculate the impact speed of an object dropped from {h} meters.",
        f"What speed does a {m} kg ball reach after free-falling {h} m from rest?",
        f"An object falls from rest through a vertical distance of {h} m. Determine its final speed.",
    ]


def _spring_templates(p: dict) -> list[str]:
    k = p.get("k_Nm", 100)
    m = p.get("mass_kg", 2)
    x = p.get("x_m", 0.1)
    return [
        f"A {m} kg mass on a spring (k = {k} N/m) is displaced {x} m. What is the period of oscillation?",
        f"Find the oscillation period of a {m}-kg block attached to a spring with constant {k} N/m.",
        f"Calculate T for a mass-spring system: m = {m} kg, k = {k} N/m.",
        f"What is the period of simple harmonic motion for a {m} kg mass on a {k} N/m spring?",
    ]


def _pendulum_templates(p: dict) -> list[str]:
    L = p.get("L_m", 1.0)
    return [
        f"A simple pendulum has length {L} m. What is its period (small angle)?",
        f"Find the period of a {L}-m pendulum under small oscillations.",
        f"Calculate the period of a simple pendulum of length {L} m near Earth's surface.",
        f"What is the oscillation period for a pendulum that is {L} m long?",
    ]


def _circuit_templates(p: dict) -> list[str]:
    V = p.get("V_battery", 12)
    R1 = p.get("R1_ohm", 10)
    R2 = p.get("R2_ohm", 20)
    config = p.get("config", "series")
    return [
        f"Two resistors ({R1} Ω and {R2} Ω) are connected in {config} to a {V} V battery. Find the current.",
        f"Calculate the total current from a {V} V source through {R1} Ω and {R2} Ω resistors in {config}.",
        f"What current flows through a {config} circuit with R₁ = {R1} Ω, R₂ = {R2} Ω, and V = {V} V?",
        f"A {V}-V battery drives current through two {config} resistors of {R1} Ω and {R2} Ω. What is the current?",
    ]


def _wave_templates(p: dict) -> list[str]:
    f = p.get("f_Hz", 5)
    wl = p.get("wavelength_m", 0.5)
    return [
        f"A wave has frequency {f} Hz and wavelength {wl} m. What is its speed?",
        f"Find the speed of a wave with f = {f} Hz and λ = {wl} m.",
        f"Calculate v for a wave: frequency = {f} Hz, wavelength = {wl} m.",
        f"What is the propagation speed of a wave at {f} Hz with a {wl}-m wavelength?",
    ]


def _thermal_templates(p: dict) -> list[str]:
    m = p.get("mass_kg", 2)
    c = p.get("c_JkgK", 900)
    dT = p.get("dT_K", 50)
    return [
        f"How much heat is needed to raise the temperature of {m} kg of a substance (c = {c} J/kg·K) by {dT} K?",
        f"Calculate Q for heating {m} kg with specific heat {c} J/kg·K through a {dT} K change.",
        f"Find the thermal energy required: m = {m} kg, c = {c} J/(kg·K), ΔT = {dT} K.",
        f"What energy input raises {m} kg of material (specific heat {c}) by {dT} degrees?",
    ]


_TEMPLATE_REGISTRY = {
    "inclined_plane": _inclined_plane_templates,
    "projectile": _projectile_templates,
    "collision": _collision_templates,
    "free_fall": _free_fall_templates,
    "spring": _spring_templates,
    "pendulum": _pendulum_templates,
    "circuit": _circuit_templates,
    "wave": _wave_templates,
    "thermal": _thermal_templates,
}


def paraphrase_question(family: str, params: dict) -> str:
    templates_fn = _TEMPLATE_REGISTRY.get(family)
    if templates_fn:
        templates = templates_fn(params)
        return random.choice(templates)
    return None
