"""🎨 Visualize H2R retargeting results with viser.

Usage:
    cd omnidexgrasp
    conda activate omnidexgrasp
    python scripts/vis_dexgrasp.py
    python scripts/vis_dexgrasp.py --output ../out --assets ../assets/robo --port 8080
"""

import sys
from pathlib import Path

# Add project root to path so human2robo can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import trimesh
import torch
import viser


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
    parser.add_argument("--output",    default="../out",         help="Output directory")
    parser.add_argument("--assets",    default="../assets/robo", help="Robot hand assets")
    parser.add_argument("--mano-root", default="../assests/mano", help="MANO assets root")
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

    task_names = list(tasks.keys())
    hand_types = [ht for ht in ["inspire", "wuji", "shadow"]
                  if ht in tasks[task_names[0]]]

    state: dict = {"task": task_names[0], "hand": hand_types[0] if hand_types else ""}

    task_dd = server.gui.add_dropdown("Task",      options=task_names, initial_value=task_names[0])
    hand_dd = server.gui.add_dropdown("Hand Type", options=hand_types, initial_value=hand_types[0])
    info_md = server.gui.add_markdown("*Select task and hand type*")

    def refresh():
        task_name = state["task"]
        hand_type = state["hand"]
        robo      = tasks[task_name]

        server.scene.reset()

        # Object mesh (in object frame = at origin)
        mesh_path = output_dir / task_name / "scaled_mesh.obj"
        if mesh_path.exists():
            server.scene.add_mesh_trimesh("object", trimesh.load(str(mesh_path)))

        if hand_type in robo:
            # Final robot hand (support old {"init":…,"final":…} format)
            entry = robo[hand_type]
            final_pose = entry["final"] if isinstance(entry, dict) else entry
            final_mesh = get_hand_mesh(hand_type, final_pose, assets_root)
            if final_mesh is not None:
                final_mesh.visual.vertex_colors = [0, 100, 255, 80]  # blue, semi-transparent
                server.scene.add_mesh_trimesh("hand_final", final_mesh)

            # MANO hand mesh in obj_cam frame via dataloader
            task_data = dataloader.load(task_name)
            if task_data is not None:
                mano_verts = task_data.mano_verts_obj[0].cpu().numpy()
                mano_faces = dataloader.mano_faces.cpu().numpy()
                mano_mesh  = trimesh.Trimesh(vertices=mano_verts, faces=mano_faces)
                mano_mesh.visual.vertex_colors = [0, 200, 0, 80]   # green, semi-transparent
                server.scene.add_mesh_trimesh("hand_mano", mano_mesh)

            pose_str = ", ".join(f"{v:.3f}" for v in final_pose)
            info_md.content = (
                f"**{task_name}** | `{hand_type}`\n\n"
                f"🟢 MANO &nbsp; 🔵 robot\n\n"
                f"**dex_pose** ({len(final_pose)} dims):\n\n"
                f"```\n{pose_str}\n```"
            )
        else:
            info_md.content = f"⚠️ No {hand_type} result"

    @task_dd.on_update
    def _(e): state["task"] = e.target.value; refresh()

    @hand_dd.on_update
    def _(e): state["hand"] = e.target.value; refresh()

    refresh()
    print("🎨 Ready. Press Ctrl+C to exit.")
    import time
    while True: time.sleep(0.5)


if __name__ == "__main__":
    main()
