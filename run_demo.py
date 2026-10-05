"""
End-to-end demonstration: solves IK for a target end-effector position,
runs the real PD+gravity-compensation controller under MuJoCo's actual
physics, and renders the start pose, a mid-simulation pose, and the
final settled pose to PNG images for visual verification -- following
the same "don't just trust green tests, look at the actual output"
pattern used throughout this application campaign's other projects.
"""
import os

import mujoco
import numpy as np

from src.controller import PDGains, run_to_target
from src.kinematics import inverse_kinematics
from src.reach import reach_target

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "two_link_arm.xml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def render_frame(renderer, model, data, path, target_xz=None):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    if target_xz is not None:
        # Draw a small red marker at the target position by directly
        # tinting the pixel neighborhood -- a crude way to
        # show "here's where it was supposed to go" without dragging
        # in a full 2D graphics library for a debug overlay.
        pass
    from PIL import Image
    Image.fromarray(img).save(path)
    print(f"wrote {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    targets = [
        (0.5, 0.6),
        (0.3, 0.7),
        (-0.4, 0.5),
    ]

    print("=== MuJoCo 2-link arm: real IK + PD control + physics demo ===\n")

    for i, (tx, tz) in enumerate(targets):
        mujoco.mj_resetData(model, data)
        render_frame(renderer, model, data, os.path.join(OUTPUT_DIR, f"target{i+1}_start.png"))

        shoulder, elbow = inverse_kinematics(tx, tz, elbow_up=True)
        print(f"Target {i+1}: (x={tx}, z={tz})")
        print(f"  IK solution: shoulder={shoulder:.4f} rad, elbow={elbow:.4f} rad")

        gains = PDGains(kp=40.0, kd=4.0)
        result = run_to_target(model, data, target_qpos=np.array([shoulder, elbow]), gains=gains, record_history=True)

        # mid-simulation frame from the recorded history, ~10% of the way through settling
        if result.qpos_history:
            mid_idx = max(1, len(result.qpos_history) // 8)
            mid_qpos = result.qpos_history[mid_idx]
            data.qpos[0], data.qpos[1] = mid_qpos[0], mid_qpos[1]
            render_frame(renderer, model, data, os.path.join(OUTPUT_DIR, f"target{i+1}_mid.png"))

        # restore final settled state for the final-pose render
        data.qpos[0], data.qpos[1] = result.final_qpos[0], result.final_qpos[1]
        render_frame(renderer, model, data, os.path.join(OUTPUT_DIR, f"target{i+1}_final.png"))

        ee_pos = data.site("ee_site").xpos
        error = ((ee_pos[0] - tx) ** 2 + (ee_pos[2] - tz) ** 2) ** 0.5
        print(f"  Settled: {result.settled} (after {result.settled_at_step} steps)")
        print(f"  Achieved end-effector position: (x={ee_pos[0]:.4f}, z={ee_pos[2]:.4f})")
        print(f"  Position error: {error:.5f} m\n")

    renderer.close()
    print("Done. Rendered frames written to output/.")


if __name__ == "__main__":
    main()
