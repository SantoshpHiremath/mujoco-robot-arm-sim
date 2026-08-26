"""
Ties kinematics and control together: given a target end-effector
(x, z) position, solve inverse kinematics for the required joint
angles, then run the PD controller in real MuJoCo simulation to drive
the arm there, and report the actual final end-effector error --
measured from the simulator's own state, not assumed from the IK
solution.

This distinction matters and is checked explicitly: IK tells you which
joint angles *should* produce the target position; only actually
running the simulated dynamics with a real controller confirms the
arm actually gets there under real (if simplified) physics, with real
mass, damping, and gravity acting on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .controller import PDGains, run_to_target
from .kinematics import BASE_HEIGHT, Point2D, forward_kinematics, inverse_kinematics


@dataclass
class ReachResult:
    target: Point2D
    achieved: Point2D
    position_error: float
    settled: bool
    n_steps: int


def reach_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_x: float,
    target_z: float,
    gains: PDGains = PDGains(kp=40.0, kd=4.0),
    elbow_up: bool = True,
) -> ReachResult:
    """Solves IK for (target_x, target_z), resets the simulation to a
    neutral pose, runs the PD controller toward the IK solution, and
    reports the actual achieved end-effector position (read from
    MuJoCo's own site position, not recomputed from forward_kinematics
    -- so this is a genuine check of the simulator's behavior, not a
    restatement of the kinematics math)."""
    shoulder_target, elbow_target = inverse_kinematics(target_x, target_z, elbow_up=elbow_up)

    mujoco.mj_resetData(model, data)
    result = run_to_target(
        model, data,
        target_qpos=np.array([shoulder_target, elbow_target]),
        gains=gains,
    )

    mujoco.mj_forward(model, data)
    ee_pos = data.site("ee_site").xpos
    achieved = Point2D(x=float(ee_pos[0]), z=float(ee_pos[2]))
    target = Point2D(x=target_x, z=target_z)
    error = ((achieved.x - target.x) ** 2 + (achieved.z - target.z) ** 2) ** 0.5

    return ReachResult(
        target=target,
        achieved=achieved,
        position_error=error,
        settled=result.settled,
        n_steps=result.n_steps_run,
    )
