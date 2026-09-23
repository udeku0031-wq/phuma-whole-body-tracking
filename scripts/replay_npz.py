"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
import sys
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
motion_source = parser.add_mutually_exclusive_group(required=True)
motion_source.add_argument("--registry_name", type=str, help="The name of the wandb motion registry.")
motion_source.add_argument("--motion_file", type=str, help="Path to a local WBT motion .npz file.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum replay steps before exiting.")
parser.add_argument("--progress_interval", type=int, default=500, help="Print replay progress every N steps.")
parser.add_argument("--start_frame", type=int, default=0, help="First motion frame to replay (inclusive).")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument(
    "--end_frame_exclusive",
    type=int,
    default=None,
    help="Frame at which replay wraps (exclusive). Defaults to the motion length.",
)
parser.add_argument("--video_output", type=str, default=None, help="Optional output .mp4 path for headless replay.")
parser.add_argument("--video_fps", type=float, default=30.0, help="Output video FPS when --video_output is set.")
parser.add_argument("--video_width", type=int, default=1280, help="Output video width when --video_output is set.")
parser.add_argument("--video_height", type=int, default=720, help="Output video height when --video_output is set.")
parser.add_argument(
    "--camera_prim_path",
    type=str,
    default="/World/PresentationCamera",
    help="USD camera prim used for --video_output.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(2.2, -2.8, 1.25),
    metavar=("X", "Y", "Z"),
    help="Camera offset from the robot root.",
)
parser.add_argument(
    "--camera_lookat_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.25),
    metavar=("X", "Y", "Z"),
    help="Camera target offset from the robot root.",
)
parser.add_argument(
    "--camera_eye_world",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Fixed world-space camera position. Overrides --camera_eye when set.",
)
parser.add_argument(
    "--camera_target_world",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Fixed world-space camera target. Overrides root-follow lookat when set.",
)
parser.add_argument(
    "--camera_warmup_frames",
    type=int,
    default=8,
    help="Rendered warmup frames before writing video, useful for texture/camera initialization.",
)
parser.add_argument(
    "--physics_step_for_render",
    action="store_true",
    default=False,
    help=(
        "After writing each replay pose, advance one physics step before recording. "
        "This is a fallback for headless RTX paths that render the ground but do not "
        "refresh articulation visuals from direct state writes."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video_output is not None:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import cv2
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


class Mp4Recorder:
    """Capture a USD camera render product and write frames to an mp4."""

    def __init__(self, output_path: str, camera_prim_path: str, width: int, height: int, fps: float):
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        if not camera_prim.IsValid():
            camera_prim = stage.DefinePrim(camera_prim_path, "Camera")
        UsdGeom.Camera(camera_prim)
        self._render_product = rep.create.render_product(camera_prim_path, resolution=(width, height))
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annotator.attach([self._render_product])
        self.frames_written = 0

    def write_frame(self):
        rgb_data = self._annotator.get_data()
        frame = np.asarray(rgb_data)
        if frame.size == 0:
            return
        if frame.ndim != 3 or frame.shape[2] < 3:
            return
        frame = frame[:, :, :3]
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames_written += 1

    def close(self):
        self._writer.release()


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    recorder = None

    if args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Downloaded motion artifact: {registry_name}", flush=True)
    else:
        motion_file = args_cli.motion_file
        print(f"[INFO]: Using local motion file: {motion_file}", flush=True)
    print(f"[INFO]: Motion file: {motion_file}", flush=True)

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    motion_length = int(motion.time_step_total)
    start_frame = args_cli.start_frame
    end_frame_exclusive = args_cli.end_frame_exclusive
    if end_frame_exclusive is None:
        end_frame_exclusive = motion_length
    if not 0 <= start_frame < end_frame_exclusive <= motion_length:
        raise ValueError(
            "Invalid replay frame range "
            f"[{start_frame}, {end_frame_exclusive}) for motion length {motion_length}; "
            "expected 0 <= start_frame < end_frame_exclusive <= motion length."
        )
    print(
        f"[INFO]: Replay started. Motion steps: {motion_length}. "
        f"Frame range: [{start_frame}, {end_frame_exclusive}). "
        "Press Ctrl+C to stop, or use --max_steps for a finite test.",
        flush=True,
    )
    if args_cli.video_output is not None:
        recorder = Mp4Recorder(
            args_cli.video_output,
            camera_prim_path=args_cli.camera_prim_path,
            width=args_cli.video_width,
            height=args_cli.video_height,
            fps=args_cli.video_fps,
        )
        print(f"[INFO]: Recording replay video to: {args_cli.video_output}", flush=True)
    time_steps = torch.full((scene.num_envs,), start_frame - 1, dtype=torch.long, device=sim.device)
    replay_steps = 0
    target_steps = args_cli.max_steps
    if target_steps is None and recorder is not None:
        target_steps = end_frame_exclusive - start_frame
    camera_eye = np.asarray(args_cli.camera_eye, dtype=np.float64)
    camera_lookat_offset = np.asarray(args_cli.camera_lookat_offset, dtype=np.float64)
    camera_eye_world = (
        None if args_cli.camera_eye_world is None else np.asarray(args_cli.camera_eye_world, dtype=np.float64)
    )
    camera_target_world = (
        None if args_cli.camera_target_world is None else np.asarray(args_cli.camera_target_world, dtype=np.float64)
    )

    # Simulation loop
    try:
        while simulation_app.is_running():
            replay_steps += 1
            time_steps += 1
            reset_ids = time_steps >= end_frame_exclusive
            time_steps[reset_ids] = start_frame

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
            root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
            scene.write_data_to_sim()

            if camera_target_world is None:
                pos_lookat = root_states[0, :3].cpu().numpy() + camera_lookat_offset
            else:
                pos_lookat = camera_target_world
            if camera_eye_world is None:
                pos_eye = pos_lookat + camera_eye
            else:
                pos_eye = camera_eye_world
            sim.set_camera_view(pos_eye, pos_lookat, camera_prim_path=args_cli.camera_prim_path)
            if args_cli.physics_step_for_render:
                # Some Isaac/RTX headless combinations do not refresh articulation
                # visuals after direct state writes unless PhysX advances once.
                sim.step(render=True)
            else:
                # Teleporting articulation state for replay needs a render flush.
                # With Fabric enabled this also updates Hydra; without Fabric it
                # may still need --physics_step_for_render on some installations.
                sim.forward()
                sim.render()
            scene.update(sim_dt)

            if recorder is not None and replay_steps > args_cli.camera_warmup_frames:
                recorder.write_frame()

            if args_cli.progress_interval > 0 and replay_steps % args_cli.progress_interval == 0:
                print(f"[INFO]: Replayed {replay_steps} steps", flush=True)
            if target_steps is not None and replay_steps >= target_steps + args_cli.camera_warmup_frames:
                print(f"[INFO]: Reached replay target of {target_steps} recorded steps. Exiting.", flush=True)
                break
    finally:
        if recorder is not None:
            recorder.close()
            print(
                f"[INFO]: Saved {recorder.frames_written} frame(s) to {recorder.output_path}",
                flush=True,
            )


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    if args_cli.disable_fabric:
        sim_cfg.use_fabric = False
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
