"""🎨 Visualize H2R retargeting results with viser.

Usage:
    cd omnidexgrasp
    conda activate omnidexgrasp
    python scripts/vis_dexgrasp.py
    python scripts/vis_dexgrasp.py --output ../out --assets ../assets/robo --port 8080
"""

import sys
import math
from pathlib import Path

# Add project root to path so human2robo can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import trimesh
import torch
import viser

# 90° CCW around X: fixes up-direction so the grasp faces forward
_ROT90X = (math.sqrt(2) / 2, math.sqrt(2) / 2, 0.0, 0.0)  # wxyz


def load_tasks(output_dir: Path) -> dict[str, dict]:
    """Load all tasks that have robo.json."""
    return {
        d.name: json.load(open(d / "robo.json"))
        for d in sorted(output_dir.iterdir())
        if d.is_dir() and (d / "robo.json").exists()
    }


def get_hand_mesh(hand_type: str, dex_pose: list, assets_root: Path) -> trimesh.Trimesh | None:
    """Get robot hand trimesh from forward kinematics (fingertip link meshes excluded)."""
    try:
        from human2robo.models import HAND_MODELS
        model = HAND_MODELS[hand_type](assets_root=assets_root, device="cpu", use_convex=False)
        out = model(torch.tensor([dex_pose], dtype=torch.float32), include_fingertip_mesh=False)
        return trimesh.Trimesh(
            vertices=out["vertices"][0].detach().numpy(),
            faces=out["faces"].numpy(),
        )
    except Exception as e:
        print(f"⚠️  [{hand_type}] FK failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Visualize H2R dexterous grasp results")
    parser.add_argument("--output",    default="../out",          help="Output directory")
    parser.add_argument("--assets",    default="../assets/robo",  help="Robot hand assets")
    parser.add_argument("--mano-root", default="../assets/mano", help="MANO assets root")
    parser.add_argument("--port",      type=int, default=8080)
    args = parser.parse_args()

    output_dir  = Path(args.output)
    assets_root = Path(args.assets)
    tasks = load_tasks(output_dir)

    if not tasks:
        print("❌ No tasks with robo.json found!"); return

    from human2robo.dataloader import RetargetDataLoader
    dataloader = RetargetDataLoader(
        output_dir=output_dir,
        n_obj_pts=1,           # minimal; obj_pc not needed for visualization
        device="cpu",
        mano_assets_root=args.mano_root,
    )

    server = viser.ViserServer(port=args.port)
    print(f"🌐 Viser: http://localhost:{args.port} | {len(tasks)} tasks")

    # Scene root frame rotated 90° CCW around X so objects face forward
    server.scene.add_frame("/scene", wxyz=_ROT90X, show_axes=False)

    task_names = list(tasks.keys())
    hand_types = [ht for ht in ["inspire", "wuji", "shadow"]
                  if ht in tasks[task_names[0]]]

    state: dict = {"task": task_names[0], "hand": hand_types[0] if hand_types else "", "show_init": True}
    handles: dict = {}

    task_dd   = server.gui.add_dropdown("Task",      options=task_names, initial_value=task_names[0])
    hand_dd   = server.gui.add_dropdown("Hand Type", options=hand_types, initial_value=hand_types[0])
    show_init = server.gui.add_checkbox("Show Init Hand", initial_value=True)
    info_md   = server.gui.add_markdown("*Select task and hand type*")

    def refresh():
        task_name = state["task"]
        hand_type = state["hand"]
        robo      = tasks[task_name]

        # Remove previous mesh handles (keep /scene frame)
        for h in handles.values():
            h.remove()
        handles.clear()

        # Object mesh under /scene so it inherits the X rotation
        mesh_path = output_dir / task_name / "scaled_mesh.obj"
        if mesh_path.exists():
            handles["object"] = server.scene.add_mesh_trimesh("/scene/object", trimesh.load(str(mesh_path)))

        if hand_type in robo:
            entry = robo[hand_type]
            init_pose  = entry["init"]  if isinstance(entry, dict) else entry
            final_pose = entry["final"] if isinstance(entry, dict) else entry

            # Init robot hand (orange) — before optimization
            if state["show_init"]:
                init_mesh = get_hand_mesh(hand_type, init_pose, assets_root)
                if init_mesh is not None:
                    init_mesh.visual.vertex_colors = [255, 140, 0, 80]  # orange, semi-transparent
                    handles["hand_init"] = server.scene.add_mesh_trimesh("/scene/hand_init", init_mesh)

            # Final robot hand (blue) — after optimization
            final_mesh = get_hand_mesh(hand_type, final_pose, assets_root)
            if final_mesh is not None:
                final_mesh.visual.vertex_colors = [0, 100, 255, 80]  # blue, semi-transparent
                handles["hand_final"] = server.scene.add_mesh_trimesh("/scene/hand_final", final_mesh)

            # MANO hand mesh in obj_cam frame via dataloader
            task_data = dataloader.load(task_name)
            if task_data is not None:
                mano_verts = task_data.mano_verts_obj[0].cpu().numpy()
                mano_faces = dataloader.mano_faces.cpu().numpy()
                mano_mesh  = trimesh.Trimesh(vertices=mano_verts, faces=mano_faces)
                mano_mesh.visual.vertex_colors = [0, 200, 0, 80]   # green, semi-transparent
                handles["hand_mano"] = server.scene.add_mesh_trimesh("/scene/hand_mano", mano_mesh)

            pose_str = ", ".join(f"{v:.3f}" for v in final_pose)
            legend = "🟠 init &nbsp; 🔵 final &nbsp; 🟢 MANO" if state["show_init"] else "🔵 final &nbsp; 🟢 MANO"
            info_md.content = (
                f"**{task_name}** | `{hand_type}`\n\n"
                f"{legend}\n\n"
                f"**dex_pose final** ({len(final_pose)} dims):\n\n"
                f"```\n{pose_str}\n```"
            )
        else:
            info_md.content = f"⚠️ No {hand_type} result"

    @task_dd.on_update
    def _(e): state["task"] = e.target.value; refresh()

    @hand_dd.on_update
    def _(e): state["hand"] = e.target.value; refresh()

    @show_init.on_update
    def _(e): state["show_init"] = e.target.value; refresh()

    refresh()
    print("🎨 Ready. Press Ctrl+C to exit.")
    import time
    while True: time.sleep(0.5)


if __name__ == "__main__":
    main()
