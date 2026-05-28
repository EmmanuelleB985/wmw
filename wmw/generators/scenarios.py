from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any

from wmw.schemas.models import (
    PhysicalObject, Relation, Force, State, Transition,
    ResultingState, Answer, Metadata, Trace,
)


@dataclass
class ScenarioSpec:
    family: str
    params: dict[str, Any]
    state: State
    transition: Transition
    result: ResultingState
    answer: Answer
    question: str
    difficulty: str = "medium"
    task_type: str = "transition_prediction"


def _rand(lo: float, hi: float, decimals: int = 1) -> float:
    return round(random.uniform(lo, hi), decimals)

def _choice(items):
    return random.choice(items)

G = 9.8


def inclined_plane() -> ScenarioSpec:
    theta = _rand(15, 60)
    mass = _rand(1, 20)
    mu = _choice([0.0, _rand(0.05, 0.4)])
    frictionless = mu == 0.0
    g_parallel = round(G * math.sin(math.radians(theta)), 2)
    g_perp = round(G * math.cos(math.radians(theta)), 2)
    friction_force = round(mu * mass * g_perp, 2) if not frictionless else 0.0
    net_accel = round(g_parallel - friction_force / mass, 2) if mass > 0 else 0
    direction = "down the incline" if net_accel > 0 else "up the incline (or stationary)"

    assumptions = ["ideal rigid body"]
    if frictionless:
        assumptions.append("frictionless surface")

    return ScenarioSpec(
        family="inclined_plane",
        params={"theta": theta, "mass": mass, "mu": mu},
        state=State(
            objects=[PhysicalObject("block", {"mass": mass, "mass_unit": "kg", "shape": "rectangular"})],
            relations=[Relation("on", ("block", "incline"))],
            forces=[
                Force("gravity", "block", "downward", mass * G, "N"),
                Force("normal", "block", f"perpendicular to incline surface", mass * g_perp, "N"),
            ] + ([Force("friction", "block",
                        "up the incline" if net_accel > 0 else "down the incline",
                        friction_force, "N")] if not frictionless else []),
            variables={"incline_angle_deg": theta, "mu": mu, "g": G},
            assumptions=assumptions,
        ),
        transition=Transition(
            rule="Newton's second law along the incline",
            effect=f"Block accelerates {direction} at {abs(net_accel)} m/s²",
            equation=f"a = g·sin(θ) - μ·g·cos(θ) = {abs(net_accel)} m/s²",
            evidence=[f"incline angle = {theta}°", f"mass = {mass} kg", f"μ = {mu}"],
        ),
        result=ResultingState(
            predicted_change=f"Block accelerates {direction}",
            new_variables={"acceleration_m_s2": net_accel, "direction": direction},
        ),
        answer=Answer(
            value=f"{abs(net_accel)} m/s²",
            unit="m/s²",
            explanation=f"Net acceleration {direction}: a = g·sin({theta}°) - μ·g·cos({theta}°) = {abs(net_accel)} m/s²",
        ),
        question=f"A {mass} kg block is on a {'frictionless' if frictionless else f'rough (μ={mu})'} incline at {theta}°. What is the acceleration?",
    )


def projectile() -> ScenarioSpec:
    v0 = _rand(5, 50)
    angle = _rand(20, 70)
    v0x = round(v0 * math.cos(math.radians(angle)), 2)
    v0y = round(v0 * math.sin(math.radians(angle)), 2)
    t_peak = round(v0y / G, 2)
    h_max = round(v0y**2 / (2 * G), 2)
    r = round(v0**2 * math.sin(math.radians(2 * angle)) / G, 2)

    return ScenarioSpec(
        family="projectile",
        params={"v0": v0, "angle": angle},
        state=State(
            objects=[PhysicalObject("projectile", {"mass": "m", "shape": "point"})],
            relations=[Relation("above", ("projectile", "ground"))],
            forces=[Force("gravity", "projectile", "downward", "m·g", "N")],
            variables={"v0": v0, "launch_angle_deg": angle, "v0x": v0x, "v0y": v0y, "g": G},
            assumptions=["no air resistance", "flat ground", "point mass"],
        ),
        transition=Transition(
            rule="Kinematic equations under constant gravitational acceleration",
            effect=f"Horizontal velocity constant at {v0x} m/s; vertical velocity decreases at {G} m/s²",
            equation=f"h_max = v0y²/(2g) = {h_max} m; R = v0²·sin(2θ)/g = {r} m",
            evidence=[f"v0 = {v0} m/s", f"angle = {angle}°", "no air resistance"],
        ),
        result=ResultingState(
            predicted_change=f"Projectile reaches max height {h_max} m at t={t_peak} s, range {r} m",
            new_variables={"h_max_m": h_max, "range_m": r, "t_peak_s": t_peak},
        ),
        answer=Answer(value=h_max, unit="m", explanation=f"Maximum height = {h_max} m"),
        question=f"A projectile is launched at {v0} m/s at {angle}°. What is the maximum height?",
        task_type="transition_prediction",
    )


def collision() -> ScenarioSpec:
    m1 = _rand(1, 10)
    m2 = _rand(1, 10)
    v1 = _rand(2, 20)
    v2 = _rand(-10, 10)
    elastic = _choice([True, False])

    if elastic:
        v1f = round(((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2), 2)
        v2f = round(((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2), 2)
        ctype = "elastic"
    else:
        vf = round((m1 * v1 + m2 * v2) / (m1 + m2), 2)
        v1f = vf
        v2f = vf
        ctype = "perfectly inelastic"

    return ScenarioSpec(
        family="collision",
        params={"m1": m1, "m2": m2, "v1": v1, "v2": v2, "elastic": elastic},
        state=State(
            objects=[
                PhysicalObject("object_A", {"mass": m1, "mass_unit": "kg"}),
                PhysicalObject("object_B", {"mass": m2, "mass_unit": "kg"}),
            ],
            relations=[Relation("adjacent", ("object_A", "object_B"))],
            forces=[],
            variables={"v1_initial": v1, "v2_initial": v2, "collision_type": ctype},
            assumptions=[f"{ctype} collision", "1D motion", "no external forces"],
        ),
        transition=Transition(
            rule="Conservation of momentum" + (" and kinetic energy" if elastic else ""),
            effect=f"After collision: A moves at {v1f} m/s, B moves at {v2f} m/s",
            equation=f"m1·v1 + m2·v2 = m1·v1' + m2·v2'",
            evidence=[f"m1={m1} kg, v1={v1} m/s", f"m2={m2} kg, v2={v2} m/s"],
        ),
        result=ResultingState(
            predicted_change=f"Object A: {v1f} m/s, Object B: {v2f} m/s",
            new_variables={"v1_final": v1f, "v2_final": v2f},
        ),
        answer=Answer(value=v1f, unit="m/s", explanation=f"Final velocity of A = {v1f} m/s"),
        question=f"Object A ({m1} kg, {v1} m/s) collides with B ({m2} kg, {v2} m/s) in a {ctype} collision. Final velocity of A?",
    )


def pulley() -> ScenarioSpec:
    m1 = _rand(2, 15)
    m2_offset = _rand(1, 8)
    m2 = round(m1 + m2_offset, 1)
    accel = round((m2 - m1) * G / (m1 + m2), 2)
    tension = round(2 * m1 * m2 * G / (m1 + m2), 2)

    return ScenarioSpec(
        family="pulley",
        params={"m1": m1, "m2": m2},
        state=State(
            objects=[
                PhysicalObject("mass_1", {"mass": m1, "mass_unit": "kg", "side": "left"}),
                PhysicalObject("mass_2", {"mass": m2, "mass_unit": "kg", "side": "right"}),
                PhysicalObject("pulley", {"type": "ideal", "massless": True, "frictionless": True}),
                PhysicalObject("rope", {"massless": True, "inextensible": True}),
            ],
            relations=[
                Relation("connected", ("mass_1", "rope")),
                Relation("connected", ("mass_2", "rope")),
                Relation("on", ("rope", "pulley")),
            ],
            forces=[
                Force("gravity", "mass_1", "downward", m1 * G, "N"),
                Force("gravity", "mass_2", "downward", m2 * G, "N"),
                Force("tension", "mass_1", "upward", tension, "N"),
                Force("tension", "mass_2", "upward", tension, "N"),
            ],
            variables={"g": G},
            assumptions=["massless frictionless pulley", "massless inextensible rope"],
        ),
        transition=Transition(
            rule="Atwood machine: a = (m2-m1)g/(m1+m2)",
            effect=f"Heavier mass (m2={m2} kg) descends, lighter mass (m1={m1} kg) ascends at a={accel} m/s²",
            equation=f"a = ({m2}-{m1})·{G}/({m1}+{m2}) = {accel} m/s²",
            evidence=[f"m1={m1} kg", f"m2={m2} kg", "ideal pulley"],
        ),
        result=ResultingState(
            predicted_change=f"System accelerates at {accel} m/s², tension = {tension} N",
            new_variables={"acceleration_m_s2": accel, "tension_N": tension},
        ),
        answer=Answer(value=accel, unit="m/s²", explanation=f"Atwood acceleration = {accel} m/s²"),
        question=f"An Atwood machine has masses {m1} kg and {m2} kg. What is the acceleration?",
    )


def spring() -> ScenarioSpec:
    k = _rand(50, 500, 0)
    x = _rand(0.01, 0.3, 3)
    mass = _rand(0.5, 10)
    pe = round(0.5 * k * x**2, 3)
    force = round(k * x, 2)
    direction = _choice(["compressed", "stretched"])

    return ScenarioSpec(
        family="spring",
        params={"k": k, "x": x, "mass": mass, "direction": direction},
        state=State(
            objects=[
                PhysicalObject("spring", {"k": k, "k_unit": "N/m", "natural_length": "L₀"}),
                PhysicalObject("block", {"mass": mass, "mass_unit": "kg"}),
            ],
            relations=[Relation("attached", ("block", "spring"))],
            forces=[Force("spring", "block",
                          "toward equilibrium" if direction == "stretched" else "away from wall",
                          force, "N")],
            variables={"displacement_m": x, "k_N_per_m": k, "direction": direction},
            assumptions=["ideal spring (Hooke's law)", "horizontal surface", "no friction"],
        ),
        transition=Transition(
            rule="Hooke's law: F = -kx; PE = ½kx²",
            effect=f"Spring exerts {force} N restoring force; PE = {pe} J",
            equation=f"F = {k}·{x} = {force} N",
            evidence=[f"k = {k} N/m", f"x = {x} m"],
        ),
        result=ResultingState(
            predicted_change=f"Block accelerates toward equilibrium; converts PE to KE",
            new_variables={"spring_PE_J": pe, "spring_force_N": force},
        ),
        answer=Answer(value=pe, unit="J", explanation=f"PE = ½·{k}·{x}² = {pe} J"),
        question=f"A spring (k={k} N/m) is {direction} by {x} m with a {mass} kg block. What is the PE?",
    )


def circuit() -> ScenarioSpec:
    V = _rand(3, 24)
    R1 = _rand(10, 200)
    R2 = _rand(10, 200)
    series = _choice([True, False])

    if series:
        R_total = round(R1 + R2, 1)
        I_total = round(V / R_total, 4)
        config = "series"
    else:
        R_total = round(R1 * R2 / (R1 + R2), 2)
        I_total = round(V / R_total, 4)
        config = "parallel"

    return ScenarioSpec(
        family="circuit",
        params={"V": V, "R1": R1, "R2": R2, "series": series},
        state=State(
            objects=[
                PhysicalObject("battery", {"voltage": V, "voltage_unit": "V"}),
                PhysicalObject("R1", {"resistance": R1, "resistance_unit": "Ω"}),
                PhysicalObject("R2", {"resistance": R2, "resistance_unit": "Ω"}),
            ],
            relations=[
                Relation("connected", ("battery", "R1")),
                Relation("connected", ("R1", "R2") if series else ("battery", "R2")),
            ],
            forces=[],
            variables={"V_source": V, "R1_ohm": R1, "R2_ohm": R2, "config": config},
            assumptions=["ideal battery", "ideal resistors", "steady state"],
        ),
        transition=Transition(
            rule=f"Ohm's law with {config} resistors",
            effect=f"Total resistance {R_total} Ω, total current {I_total} A",
            equation=f"I = V/R_total = {V}/{R_total} = {I_total} A",
            evidence=[f"V = {V} V", f"R1 = {R1} Ω", f"R2 = {R2} Ω", f"config = {config}"],
        ),
        result=ResultingState(
            predicted_change=f"Current {I_total} A flows through the circuit",
            new_variables={"R_total_ohm": R_total, "I_total_A": I_total},
        ),
        answer=Answer(value=I_total, unit="A", explanation=f"I = {V}/{R_total} = {I_total} A"),
        question=f"A {V}V battery connects to {R1}Ω and {R2}Ω resistors in {config}. Total current?",
    )


def free_fall() -> ScenarioSpec:
    h = _rand(2, 100)
    t = round(math.sqrt(2 * h / G), 2)
    v_final = round(G * t, 2)

    return ScenarioSpec(
        family="free_fall",
        params={"h": h},
        state=State(
            objects=[PhysicalObject("object", {"shape": "ball", "initial_velocity": 0})],
            relations=[Relation("above", ("object", "ground"))],
            forces=[Force("gravity", "object", "downward", "m·g", "N")],
            variables={"height_m": h, "g": G, "v0": 0},
            assumptions=["no air resistance", "near Earth surface"],
        ),
        transition=Transition(
            rule="Free fall: h = ½gt², v = gt",
            effect=f"Object falls {h} m in {t} s, reaching {v_final} m/s",
            equation=f"t = √(2h/g) = √(2·{h}/{G}) = {t} s",
            evidence=[f"h = {h} m", "v0 = 0", f"g = {G} m/s²"],
        ),
        result=ResultingState(
            predicted_change=f"Object hits ground at {v_final} m/s after {t} s",
            new_variables={"t_fall_s": t, "v_final_m_s": v_final},
        ),
        answer=Answer(value=t, unit="s", explanation=f"Fall time = {t} s"),
        question=f"An object falls from {h} m. How long until it hits the ground?",
    )


def friction() -> ScenarioSpec:
    mass = _rand(1, 30)
    mu_s = _rand(0.2, 0.8)
    mu_k = round(mu_s * _rand(0.6, 0.9), 2)
    F_applied = _rand(5, 200)
    F_normal = round(mass * G, 2)
    F_static_max = round(mu_s * F_normal, 2)
    F_kinetic = round(mu_k * F_normal, 2)
    moves = F_applied > F_static_max
    net_force = round(F_applied - F_kinetic, 2) if moves else 0
    accel = round(net_force / mass, 2) if moves else 0

    return ScenarioSpec(
        family="friction",
        params={"mass": mass, "mu_s": mu_s, "mu_k": mu_k, "F_applied": F_applied},
        state=State(
            objects=[PhysicalObject("block", {"mass": mass, "mass_unit": "kg"})],
            relations=[Relation("on", ("block", "surface"))],
            forces=[
                Force("gravity", "block", "downward", mass * G, "N"),
                Force("normal", "block", "upward", F_normal, "N"),
                Force("applied", "block", "horizontal (right)", F_applied, "N"),
                Force("friction", "block",
                      "horizontal (left)" if moves else "horizontal (left, static)",
                      F_kinetic if moves else F_applied, "N"),
            ],
            variables={"mu_s": mu_s, "mu_k": mu_k, "F_applied_N": F_applied},
            assumptions=["horizontal surface", "constant applied force"],
        ),
        transition=Transition(
            rule="Compare applied force to max static friction; if exceeded, use kinetic friction",
            effect=f"{'Block moves: a = {}'.format(accel) if moves else 'Block remains stationary (F_applied ≤ F_static_max)'}",
            equation=f"F_static_max = μs·N = {F_static_max} N; F_applied = {F_applied} N",
            evidence=[f"μs = {mu_s}", f"μk = {mu_k}", f"N = {F_normal} N"],
        ),
        result=ResultingState(
            predicted_change=f"{'Accelerates at ' + str(accel) + ' m/s² to the right' if moves else 'Remains stationary'}",
            new_variables={"moves": moves, "acceleration_m_s2": accel},
        ),
        answer=Answer(
            value="yes" if moves else "no",
            explanation=f"F_applied ({F_applied} N) {'>' if moves else '≤'} F_static_max ({F_static_max} N)",
        ),
        question=f"A {mass} kg block on a surface (μs={mu_s}, μk={mu_k}) has {F_applied} N applied. Does it move?",
        task_type="static_state",
    )


def circular_motion() -> ScenarioSpec:
    mass = _rand(0.5, 10)
    v = _rand(2, 20)
    r = _rand(0.5, 10)
    ac = round(v**2 / r, 2)
    Fc = round(mass * ac, 2)

    return ScenarioSpec(
        family="circular_motion",
        params={"mass": mass, "v": v, "r": r},
        state=State(
            objects=[PhysicalObject("object", {"mass": mass, "mass_unit": "kg"})],
            relations=[],
            forces=[Force("centripetal", "object", "toward center", Fc, "N")],
            variables={"speed_m_s": v, "radius_m": r},
            assumptions=["uniform circular motion", "horizontal plane"],
        ),
        transition=Transition(
            rule="Centripetal acceleration: ac = v²/r",
            effect=f"Object moves in circle with centripetal acceleration {ac} m/s²",
            equation=f"ac = {v}²/{r} = {ac} m/s²",
            evidence=[f"v = {v} m/s", f"r = {r} m"],
        ),
        result=ResultingState(
            predicted_change=f"Centripetal force {Fc} N directed toward center",
            new_variables={"ac_m_s2": ac, "Fc_N": Fc},
        ),
        answer=Answer(value=ac, unit="m/s²", explanation=f"ac = v²/r = {ac} m/s²"),
        question=f"A {mass} kg object moves at {v} m/s in a circle of radius {r} m. Centripetal acceleration?",
    )


def pendulum() -> ScenarioSpec:
    L = _rand(0.2, 5.0)
    T = round(2 * math.pi * math.sqrt(L / G), 3)
    theta_max = _rand(5, 15)

    return ScenarioSpec(
        family="pendulum",
        params={"L": L, "theta_max": theta_max},
        state=State(
            objects=[
                PhysicalObject("bob", {"shape": "sphere"}),
                PhysicalObject("string", {"length_m": L, "massless": True}),
            ],
            relations=[Relation("attached", ("bob", "string"))],
            forces=[
                Force("gravity", "bob", "downward", "m·g", "N"),
                Force("tension", "bob", "along string toward pivot", None, "N"),
            ],
            variables={"L_m": L, "theta_max_deg": theta_max, "g": G},
            assumptions=["small angle approximation", "massless string", "no air resistance"],
        ),
        transition=Transition(
            rule="Simple pendulum period: T = 2π√(L/g)",
            effect=f"Pendulum oscillates with period {T} s",
            equation=f"T = 2π√({L}/{G}) = {T} s",
            evidence=[f"L = {L} m", f"g = {G} m/s²"],
        ),
        result=ResultingState(
            predicted_change=f"Periodic motion with T = {T} s",
            new_variables={"period_s": T},
        ),
        answer=Answer(value=T, unit="s", explanation=f"T = 2π√(L/g) = {T} s"),
        question=f"A pendulum has length {L} m. What is its period (small angle)?",
    )


def lever() -> ScenarioSpec:
    F1 = _rand(10, 200)
    d1 = _rand(0.5, 3.0)
    d2 = _rand(0.5, 3.0)
    F2 = round(F1 * d1 / d2, 2)

    return ScenarioSpec(
        family="lever",
        params={"F1": F1, "d1": d1, "d2": d2},
        state=State(
            objects=[
                PhysicalObject("lever", {"type": "rigid beam"}),
                PhysicalObject("fulcrum", {}),
                PhysicalObject("load", {"force_N": F1}),
                PhysicalObject("effort", {}),
            ],
            relations=[
                Relation("supported_by", ("lever", "fulcrum")),
                Relation("on", ("load", "lever")),
            ],
            forces=[
                Force("load_weight", "lever", "downward at d1", F1, "N"),
                Force("effort", "lever", "downward at d2", F2, "N"),
            ],
            variables={"d1_m": d1, "d2_m": d2, "F1_N": F1},
            assumptions=["rigid lever", "massless beam", "static equilibrium"],
        ),
        transition=Transition(
            rule="Torque balance: F1·d1 = F2·d2",
            effect=f"Effort force = {F2} N balances the load",
            equation=f"F2 = F1·d1/d2 = {F1}·{d1}/{d2} = {F2} N",
            evidence=[f"F1 = {F1} N", f"d1 = {d1} m", f"d2 = {d2} m"],
        ),
        result=ResultingState(
            predicted_change=f"System in equilibrium with effort = {F2} N",
            new_variables={"F2_N": F2},
        ),
        answer=Answer(value=F2, unit="N", explanation=f"F2 = {F1}·{d1}/{d2} = {F2} N"),
        question=f"A lever has a {F1} N load at {d1} m from the fulcrum. Effort arm is {d2} m. Required effort force?",
    )


def buoyancy() -> ScenarioSpec:
    rho_obj = _rand(200, 2000, 0)
    rho_fluid = _choice([1000, 1025, 800, 13600])
    fluid_name = {1000: "water", 1025: "seawater", 800: "oil", 13600: "mercury"}[rho_fluid]
    V = _rand(0.001, 0.1, 4)
    F_buoy = round(rho_fluid * G * V, 2)
    F_gravity = round(rho_obj * G * V, 2)
    floats = rho_obj < rho_fluid

    return ScenarioSpec(
        family="buoyancy",
        params={"rho_obj": rho_obj, "rho_fluid": rho_fluid, "V": V},
        state=State(
            objects=[
                PhysicalObject("object", {"density_kg_m3": rho_obj, "volume_m3": V}),
                PhysicalObject("fluid", {"density_kg_m3": rho_fluid, "name": fluid_name}),
            ],
            relations=[Relation("in", ("object", "fluid"))],
            forces=[
                Force("gravity", "object", "downward", F_gravity, "N"),
                Force("buoyancy", "object", "upward", F_buoy, "N"),
            ],
            variables={"rho_object": rho_obj, "rho_fluid": rho_fluid, "V_m3": V},
            assumptions=["uniform density", "fully submerged initially"],
        ),
        transition=Transition(
            rule="Archimedes' principle: F_buoy = ρ_fluid · g · V",
            effect=f"Object {'floats (rises)' if floats else 'sinks'} in {fluid_name}",
            equation=f"F_buoy = {rho_fluid}·{G}·{V} = {F_buoy} N; F_grav = {F_gravity} N",
            evidence=[f"ρ_object = {rho_obj} kg/m³", f"ρ_fluid = {rho_fluid} kg/m³"],
        ),
        result=ResultingState(
            predicted_change=f"Object {'rises to surface' if floats else 'sinks to bottom'}",
            new_variables={"floats": floats, "F_buoyancy_N": F_buoy},
        ),
        answer=Answer(
            value="floats" if floats else "sinks",
            explanation=f"ρ_obj ({rho_obj}) {'<' if floats else '>'} ρ_fluid ({rho_fluid})",
        ),
        question=f"An object (ρ={rho_obj} kg/m³) is placed in {fluid_name} (ρ={rho_fluid}). Does it float or sink?",
        task_type="static_state",
    )


def thermal() -> ScenarioSpec:
    mass = _rand(0.1, 5.0)
    c = _choice([4186, 900, 385, 130, 2090])
    mat_name = {4186: "water", 900: "aluminum", 385: "copper", 130: "gold", 2090: "ice"}[c]
    T1 = _rand(-20, 80)
    T2 = _rand(T1 + 5, 150)
    dT = round(T2 - T1, 1)
    Q = round(mass * c * dT, 1)

    return ScenarioSpec(
        family="thermal",
        params={"mass": mass, "c": c, "T1": T1, "T2": T2, "material": mat_name},
        state=State(
            objects=[PhysicalObject(mat_name, {"mass_kg": mass, "specific_heat_J_kgK": c})],
            relations=[],
            forces=[],
            variables={"T_initial_C": T1, "T_final_C": T2, "c": c},
            assumptions=["no phase change", "no heat loss to environment"],
        ),
        transition=Transition(
            rule="Q = mcΔT",
            effect=f"{Q} J of heat {'absorbed' if Q > 0 else 'released'}",
            equation=f"Q = {mass}·{c}·{dT} = {Q} J",
            evidence=[f"mass = {mass} kg", f"c = {c} J/(kg·K)", f"ΔT = {dT} K"],
        ),
        result=ResultingState(
            predicted_change=f"Temperature changes from {T1}°C to {T2}°C",
            new_variables={"Q_J": Q, "dT_K": dT},
        ),
        answer=Answer(value=Q, unit="J", explanation=f"Q = mcΔT = {Q} J"),
        question=f"How much heat is needed to raise {mass} kg of {mat_name} from {T1}°C to {T2}°C?",
    )


def wave() -> ScenarioSpec:
    f = _rand(100, 10000, 0)
    wavelength = _rand(0.01, 3.0, 3)
    v = round(f * wavelength, 2)

    return ScenarioSpec(
        family="wave",
        params={"f": f, "wavelength": wavelength},
        state=State(
            objects=[PhysicalObject("wave", {"type": "mechanical"})],
            relations=[],
            forces=[],
            variables={"frequency_Hz": f, "wavelength_m": wavelength},
            assumptions=["constant medium"],
        ),
        transition=Transition(
            rule="Wave equation: v = fλ",
            effect=f"Wave travels at {v} m/s",
            equation=f"v = {f}·{wavelength} = {v} m/s",
            evidence=[f"f = {f} Hz", f"λ = {wavelength} m"],
        ),
        result=ResultingState(
            predicted_change=f"Wave propagates at {v} m/s",
            new_variables={"v_m_s": v},
        ),
        answer=Answer(value=v, unit="m/s", explanation=f"v = fλ = {v} m/s"),
        question=f"A wave has frequency {f} Hz and wavelength {wavelength} m. What is its speed?",
    )


def optics() -> ScenarioSpec:
    f_lens = _rand(5, 50)
    d_o = _rand(f_lens + 2, 100)
    d_i = round(1 / (1/f_lens - 1/d_o), 2)
    mag = round(-d_i / d_o, 3)
    real = d_i > 0
    inverted = mag < 0

    return ScenarioSpec(
        family="optics",
        params={"f_cm": f_lens, "d_o_cm": d_o},
        state=State(
            objects=[
                PhysicalObject("lens", {"focal_length_cm": f_lens, "type": "converging"}),
                PhysicalObject("object", {"distance_cm": d_o, "side": "left"}),
            ],
            relations=[Relation("left_of", ("object", "lens"))],
            forces=[],
            variables={"f_cm": f_lens, "d_o_cm": d_o},
            assumptions=["thin lens", "paraxial rays"],
        ),
        transition=Transition(
            rule="Thin lens equation: 1/f = 1/d_o + 1/d_i",
            effect=f"Image forms at {d_i} cm ({'real' if real else 'virtual'}, {'inverted' if inverted else 'upright'})",
            equation=f"1/{f_lens} = 1/{d_o} + 1/d_i → d_i = {d_i} cm",
            evidence=[f"f = {f_lens} cm", f"d_o = {d_o} cm"],
        ),
        result=ResultingState(
            predicted_change=f"{'Real inverted' if real and inverted else 'Virtual upright'} image at {abs(d_i)} cm",
            new_variables={"d_i_cm": d_i, "magnification": mag, "real": real, "inverted": inverted},
        ),
        answer=Answer(value=d_i, unit="cm", explanation=f"Image distance = {d_i} cm, magnification = {mag}"),
        question=f"A converging lens (f={f_lens} cm) has an object at {d_o} cm. Where is the image?",
    )


def fluid_pressure() -> ScenarioSpec:
    rho = _choice([1000, 1025, 800, 13600])
    fluid_name = {1000: "water", 1025: "seawater", 800: "oil", 13600: "mercury"}[rho]
    h = _rand(0.5, 50.0)
    P_atm = 101325
    P_gauge = round(rho * G * h, 1)
    P_abs = round(P_atm + P_gauge, 1)

    return ScenarioSpec(
        family="fluid",
        params={"rho": rho, "h": h},
        state=State(
            objects=[PhysicalObject("fluid", {"density_kg_m3": rho, "name": fluid_name})],
            relations=[],
            forces=[],
            variables={"depth_m": h, "rho_kg_m3": rho, "P_atm_Pa": P_atm, "g": G},
            assumptions=["incompressible fluid", "static fluid"],
        ),
        transition=Transition(
            rule="Hydrostatic pressure: P = P_atm + ρgh",
            effect=f"Pressure at depth {h} m = {P_abs} Pa",
            equation=f"P = {P_atm} + {rho}·{G}·{h} = {P_abs} Pa",
            evidence=[f"ρ = {rho} kg/m³", f"h = {h} m"],
        ),
        result=ResultingState(
            predicted_change=f"Absolute pressure = {P_abs} Pa, gauge pressure = {P_gauge} Pa",
            new_variables={"P_abs_Pa": P_abs, "P_gauge_Pa": P_gauge},
        ),
        answer=Answer(value=P_gauge, unit="Pa", explanation=f"Gauge pressure = ρgh = {P_gauge} Pa"),
        question=f"What is the gauge pressure at {h} m depth in {fluid_name}?",
    )


def em_induction() -> ScenarioSpec:
    N = _choice([1, 5, 10, 20, 50, 100])
    B = _rand(0.01, 2.0, 3)
    A = _rand(0.001, 0.5, 4)
    dt = _rand(0.01, 5.0, 2)
    theta_i = 0
    theta_f = _choice([90, 180])

    import math
    phi_i = round(B * A * math.cos(math.radians(theta_i)), 6)
    phi_f = round(B * A * math.cos(math.radians(theta_f)), 6)
    d_phi = round(phi_f - phi_i, 6)
    emf = round(-N * d_phi / dt, 4)

    return ScenarioSpec(
        family="em_induction",
        params={"N": N, "B": B, "A": A, "dt": dt, "theta_f": theta_f},
        state=State(
            objects=[
                PhysicalObject("coil", {"turns": N, "area_m2": A}),
                PhysicalObject("magnetic_field", {"magnitude_T": B, "direction": "perpendicular to coil"}),
            ],
            relations=[Relation("in", ("coil", "magnetic_field"))],
            forces=[],
            variables={
                "N_turns": N, "B_T": B, "A_m2": A,
                "theta_initial_deg": theta_i, "theta_final_deg": theta_f,
                "dt_s": dt,
            },
            assumptions=["uniform magnetic field", "rigid coil", "instantaneous rotation"],
        ),
        transition=Transition(
            rule="Faraday's law of electromagnetic induction: EMF = -N·dΦ/dt",
            effect=f"Coil rotates from {theta_i}° to {theta_f}° in {dt}s, inducing EMF = {abs(emf)} V",
            equation=f"EMF = -N·(Φ_f - Φ_i)/Δt = -{N}·({phi_f} - {phi_i})/{dt} = {emf} V",
            evidence=[f"N = {N} turns", f"B = {B} T", f"A = {A} m²", f"Δt = {dt} s"],
        ),
        result=ResultingState(
            predicted_change=f"EMF of {abs(emf)} V induced across coil",
            new_variables={"emf_V": emf, "d_phi_Wb": d_phi},
        ),
        answer=Answer(value=abs(emf), unit="V",
                      explanation=f"|EMF| = N·|dΦ/dt| = {abs(emf)} V"),
        question=f"A {N}-turn coil (area {A} m²) in a {B} T field rotates from {theta_i}° to {theta_f}° in {dt} s. What is the induced EMF?",
    )


SCENARIO_FAMILIES: dict[str, callable] = {
    "inclined_plane": inclined_plane,
    "projectile": projectile,
    "collision": collision,
    "pulley": pulley,
    "spring": spring,
    "circuit": circuit,
    "free_fall": free_fall,
    "friction": friction,
    "circular_motion": circular_motion,
    "pendulum": pendulum,
    "lever": lever,
    "buoyancy": buoyancy,
    "thermal": thermal,
    "wave": wave,
    "optics": optics,
    "fluid": fluid_pressure,
    "em_induction": em_induction,
}


def generate_scenario(family: str | None = None) -> ScenarioSpec:
    if family is None:
        family = random.choice(list(SCENARIO_FAMILIES.keys()))
    return SCENARIO_FAMILIES[family]()


def generate_balanced(n: int) -> list[ScenarioSpec]:
    families = list(SCENARIO_FAMILIES.keys())
    per_family = max(1, n // len(families))
    remainder = n - per_family * len(families)
    specs = []
    for fam in families:
        for _ in range(per_family):
            specs.append(generate_scenario(fam))

    for _ in range(remainder):
        specs.append(generate_scenario())
    random.shuffle(specs)
    return specs[:n]
