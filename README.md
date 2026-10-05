# mujoco-robot-arm-sim

A 2-link planar robot arm simulated in [MuJoCo](https://mujoco.org/) (DeepMind's free, open-source physics engine), with closed-form forward and inverse kinematics, a PD plus gravity-compensation joint controller, and an end-to-end IK-solve, simulate, and verify pipeline. I built it to work through robotics-simulation fundamentals: kinematics, closed-loop control under rigid-body dynamics, and cross-checking hand-written math against a physics engine's own ground truth.

## What it does

- An MJCF (MuJoCo XML) model of a 2-link arm: a fixed base, a shoulder hinge joint, an elbow hinge joint, and a tracked end-effector site.
- Closed-form forward and inverse kinematics (law-of-cosines-based), implemented independently of MuJoCo and then cross-checked against MuJoCo's simulated end-effector position.
- A joint-space PD controller with gravity compensation (feedforward `qfrc_bias`), driving the simulated arm from physics stepping (`mj_step`) rather than a kinematic "teleport to target" shortcut.
- 33 automated tests, all passing, including a dedicated cross-check test that compares the analytical kinematics against MuJoCo's simulated end-effector position for 6 different joint-angle pairs.
- Rendered PNG frames (via MuJoCo's EGL-based offscreen renderer) showing the arm's start pose and settled final pose for three different targets, in `output/`.

## Scope

The project is a small, focused 2-link planar arm in pure simulation. It uses direct PD control to a single target pose; trajectory planning, obstacle avoidance, and ROS/MoveIt integration are outside its scope, and no physical hardware is involved.

## Results

`python3 run_demo.py` runs an end-to-end IK-solve-then-simulate pipeline for three target end-effector positions, all settling to within ~1.5mm of the requested target under PD plus gravity-compensation control, and renders start, mid, and final frames via MuJoCo's EGL offscreen renderer to `output/` for visual confirmation.

## Tests

`python3 -m pytest -v` runs 33 tests (33/33 passing), covering kinematics (forward, inverse, round-trip, boundary/error cases), the PD control law in isolation, closed-loop settling behavior under simulated dynamics, a control-theory property (higher derivative gain does not increase step-response overshoot), and the analytical-vs-simulated kinematics cross-check.

## Project structure

```
models/two_link_arm.xml   MJCF model definition
src/kinematics.py         Forward/inverse kinematics (pure Python, no MuJoCo dependency)
src/controller.py         PD + gravity-compensation joint controller, simulation-run helper
src/reach.py              Ties IK + control together: target (x,z) -> simulated reach attempt
tests/                    33 tests across kinematics, controller, reach, and sim-vs-analytical cross-check
run_demo.py               End-to-end demo: solves IK, runs control, renders frames to output/
```

## Running it

```bash
pip install -r requirements.txt
python3 -m pytest -v      # run the full test suite
python3 run_demo.py       # run the IK+control+render demo, writes output/*.png
```

Offscreen rendering uses MuJoCo's EGL backend (`MUJOCO_GL=egl`). On a headless Linux machine, set `MUJOCO_GL=egl` explicitly if the default backend does not work (the OSMesa backend needs the system OSMesa libraries installed).

## Notes

Two issues came up during development, and the test suite and direct simulator introspection caught both.

**1. Sign convention for forward/inverse kinematics.** The MJCF model's hinge joints rotate about axis `"0 1 0"` (world +Y). In MuJoCo's right-handed coordinate frame, a *positive* rotation about +Y rotates a link's local +X axis *toward* -Z, the opposite of the textbook 2D-robotics-arm convention where positive angle sweeps toward +Z/up. The kinematics module was first written with the textbook sign and looked internally consistent (forward and inverse round-tripped against each other), but `tests/test_sim_matches_kinematics.py`, which cross-checks `forward_kinematics()` against MuJoCo's own simulated end-effector site position, failed for every non-zero angle pair. I fixed it by deriving the correct sign convention directly from MuJoCo's reported joint and body positions (see the docstrings in `src/kinematics.py` for the derivation).

**2. Adjacent-body self-collision.** The base, link1, and link2 bodies touch by construction (each link's capsule starts exactly where the previous joint is). MuJoCo does not automatically exclude parent/child body collisions, so without an explicit `<contact><exclude>` block the simulation picked up spurious self-contact forces every step, showing up as the arm oscillating under PD control and never settling. I diagnosed it by checking `data.ncon` (number of active contacts) at a resting pose, finding it nonzero, and inspecting which geom pairs were colliding (`link1_geom` against the unnamed base cylinder). The fix was explicit `<exclude>` entries for each adjacent body pair.

**Gravity compensation.** Plain PD control (`torque = kp * error - kd * velocity`) has an expected steady-state droop under gravity, since there is no integral term to cancel a constant disturbance, and the arm settled a few centimeters short of target. Rather than loosening test tolerances, I extended the controller with gravity (and Coriolis/centrifugal) compensation using MuJoCo's own `qfrc_bias` as a feedforward term, a standard robotics technique.

## Possible extensions

- Trajectory planning (for example minimum-jerk or spline paths) between target poses.
- Obstacle-aware planning with collision checking.
- A ROS 2 wrapper around the controller.
- A third link or a 3D arm with a Jacobian-based IK solver.
