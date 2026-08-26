"""
Forward and inverse kinematics for the two-link planar arm, computed
independently of MuJoCo's own physics engine using closed-form
trigonometry. Used both as a real robotics-fundamentals implementation
and as a cross-check: `forward_kinematics` is verified against the
simulator's own reported end-effector position in
tests/test_sim_matches_kinematics.py, so this isn't just "the math
looks right" -- it's checked against the actual physics engine's output.

Arm geometry (must match models/two_link_arm.xml):
  - Base height: 0.5 (z offset, irrelevant to the planar x-y kinematics
    below except as a constant z offset)
  - Link 1 length: 0.4
  - Link 2 length: 0.3
  - Joint 1 ("shoulder"): rotates link 1 in the x-z plane
  - Joint 2 ("elbow"): rotates link 2 relative to link 1
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LINK1_LENGTH = 0.4
LINK2_LENGTH = 0.3
BASE_HEIGHT = 0.5


@dataclass
class Point2D:
    x: float
    z: float


def forward_kinematics(shoulder_angle: float, elbow_angle: float) -> Point2D:
    """The end-effector position in the x-z plane, given the two joint
    angles (radians). This is standard 2-link planar forward
    kinematics: each link's endpoint is the previous joint's position
    plus a rotated link-length vector.

    Sign convention note: the MJCF model's hinge joints rotate about
    axis="0 1 0" (world +Y). In MuJoCo's right-handed coordinate frame,
    a positive rotation about +Y rotates the link's local +X direction
    *toward* -Z, not +Z. This was originally implemented with the
    opposite (more "intuitive" 2D-textbook) sign and was wrong -- caught
    by tests/test_sim_matches_kinematics.py, which cross-checks this
    function's output against MuJoCo's own simulated end-effector site
    position and failed for every non-zero angle pair. The minus sign
    on both z terms below is what actually matches the simulator.
    """
    x1 = LINK1_LENGTH * math.cos(shoulder_angle)
    z1 = -LINK1_LENGTH * math.sin(shoulder_angle)

    total_angle = shoulder_angle + elbow_angle
    x2 = x1 + LINK2_LENGTH * math.cos(total_angle)
    z2 = z1 - LINK2_LENGTH * math.sin(total_angle)

    return Point2D(x=x2, z=BASE_HEIGHT + z2)


class UnreachableTargetError(ValueError):
    """Raised when a requested end-effector position is farther from
    the base than the arm can physically reach, or closer than the
    minimum reach (|link1 - link2|)."""


def inverse_kinematics(target_x: float, target_z: float, elbow_up: bool = True) -> tuple[float, float]:
    """Solve for (shoulder_angle, elbow_angle) that places the
    end-effector at (target_x, target_z), using the standard 2-link
    planar IK closed-form solution (law of cosines). Raises
    UnreachableTargetError if the target is outside the arm's
    reachable workspace -- checked explicitly rather than letting the
    law-of-cosines computation silently produce a domain error or a
    wrong answer via clamping.

    elbow_up selects between the two valid elbow configurations that
    reach most targets (elbow bent one way vs. the other) -- a real
    ambiguity in 2-link IK that a robotics engineer has to resolve
    explicitly, not something with a single "correct" answer.

    Sign convention note: this must invert the same z-sign convention
    used in forward_kinematics (see that function's docstring) -- the
    hinge's positive-angle rotation moves the local x-axis toward -z in
    MuJoCo's frame for axis="0 1 0", i.e.
        x = L1*cos(s) + L2*cos(s+e)
        z = BASE - L1*sin(s) - L2*sin(s+e)
    Substituting s' = -s, e' = -e turns this into the textbook
    positive-angle-moves-toward-+z form:
        x = L1*cos(s') + L2*cos(s'+e')
        z = BASE + L1*sin(s') + L2*sin(s'+e')
    (x is unchanged since cosine is even). So this solves the standard
    closed-form 2-link IK for (s', e') using rel_z = target_z - BASE
    directly (unnegated), then negates both angles at the end to
    convert back to the simulator's actual (s, e) convention. Negating
    only one of the two angles would NOT work here, since forward_
    kinematics couples them through total_angle = s + e.
    """
    rel_z = target_z - BASE_HEIGHT
    distance = math.hypot(target_x, rel_z)

    max_reach = LINK1_LENGTH + LINK2_LENGTH
    min_reach = abs(LINK1_LENGTH - LINK2_LENGTH)

    if distance > max_reach:
        raise UnreachableTargetError(
            f"Target at distance {distance:.4f} exceeds max reach {max_reach:.4f}"
        )
    if distance < min_reach:
        raise UnreachableTargetError(
            f"Target at distance {distance:.4f} is inside min reach {min_reach:.4f} "
            f"(the arm cannot fold small enough to reach it)"
        )

    # Law of cosines for the elbow angle.
    cos_elbow = (distance**2 - LINK1_LENGTH**2 - LINK2_LENGTH**2) / (2 * LINK1_LENGTH * LINK2_LENGTH)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))  # guard only against float rounding at the boundary
    elbow_magnitude = math.acos(cos_elbow)
    elbow_angle = elbow_magnitude if elbow_up else -elbow_magnitude

    # Shoulder angle: angle to target, adjusted by the elbow's effect
    # (solved in the "textbook" +z frame, then negated back below).
    angle_to_target = math.atan2(rel_z, target_x)
    k1 = LINK1_LENGTH + LINK2_LENGTH * math.cos(elbow_angle)
    k2 = LINK2_LENGTH * math.sin(elbow_angle)
    shoulder_angle = angle_to_target - math.atan2(k2, k1)

    return -shoulder_angle, -elbow_angle
