# Coverage Map: TraceBank Physics Families

Maps the 16 seed families to standard introductory physics curriculum areas.
The claim is an extensible trace interface, not complete coverage of physics.

## Coverage Table

| Curriculum Area | Coverage | Seed Families |
|---|---|---|
| Kinematics/dynamics | **Yes** | inclined_plane, collision, projectile, free_fall, friction |
| Energy/momentum | **Yes** | collision, spring |
| Rotation/circular motion | **Partial** | circular_motion, lever (torque-lite) |
| Electrostatics/circuits | **Yes** | circuit (series/parallel resistors) |
| Magnetism/induction | **Partial** | (magnetic force stub); em_induction extension stub |
| Waves/optics | **Partial** | wave (v=fλ), optics (thin lens) |
| Pressure/fluids | **Partial** | fluid (hydrostatic pressure), buoyancy |
| Thermodynamics | **Partial** | thermal (Q=mcΔT only, no cycles) |
| Oscillations | **Yes** | pendulum, spring |
| Quantum/nuclear/relativity | **No** | excluded |

## Explicit Exclusions

- **Thermodynamic cycles**: no Carnot, Stirling, or heat-engine scenarios.
- **Electromagnetic induction**: Faraday's law scenarios not in seed set
  (but extension stub provided).
- **Quantum mechanics**: no wave-particle duality, tunneling, or energy levels.
- **Nuclear physics**: no decay, fission, or fusion.
- **Relativity**: no Lorentz transformation or relativistic mechanics.
- **Advanced fluid dynamics**: no Bernoulli flow, turbulence, or viscosity.

## Extensibility

Adding a new family requires:
1. A scenario generator function that returns a `ScenarioSpec`
2. Registration in `SCENARIO_FAMILIES` dict
3. Expected rule keywords in `_EXPECTED_RULES` (transition verifier)
4. Optional: new perturbation functions tagged to the family

The trace schema, verifier interface, and preference-pair generator
do not require modification. See `wmw/generators/scenarios.py` for
the `em_induction()` extension stub.

## Mapping to AP Physics C Topics

| AP Topic | Covered? | Notes |
|---|---|---|
| Kinematics | Yes | free_fall, projectile |
| Newton's Laws | Yes | inclined_plane, friction, pulley |
| Work, Energy, Power | Partial | spring (PE), collision (KE) |
| Linear Momentum | Yes | collision |
| Rotation | Partial | circular_motion, lever |
| Oscillations | Yes | pendulum, spring |
| Gravitation | Partial | free_fall (surface only) |
| Electric Fields | Partial | circuit (no field lines) |
| DC Circuits | Yes | circuit |
| Magnetic Fields | Stub | extension stub only |
| Electromagnetic Induction | Stub | extension stub only |
| Waves and Optics | Partial | wave, optics |
