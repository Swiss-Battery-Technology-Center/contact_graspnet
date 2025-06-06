import argparse, glob, os, sys, numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("⚠️  TensorFlow sees no GPU – running on CPU")

from contact_graspnet import config_utils
from contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet.data import load_available_input_data
from contact_graspnet.visualization_utils import visualize_grasps_o3d   # <- only 3-D

# ────────────────────────────────────────────────────────────────
def inference(
    global_config,
    checkpoint_dir,
    input_paths,
    K=None,
    local_regions=True,
    skip_border_objects=False,
    filter_grasps=True,
    segmap_id=None,
    z_range=[0.2, 1.8],
    forward_passes=1,
    save_visualization_dir=None,
):
    """Run Contact-GraspNet on each *.npy and pop up an Open3D window."""
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    saver = tf.train.Saver(save_relative_paths=True)
    sess_cfg = tf.ConfigProto()
    sess_cfg.gpu_options.allow_growth = True
    sess_cfg.allow_soft_placement = True
    sess = tf.Session(config=sess_cfg)
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode="test")

    os.makedirs("results/npz_predictions", exist_ok=True)
    if save_visualization_dir:
        os.makedirs(save_visualization_dir,   exist_ok=True)

    for p in glob.glob(input_paths):
        print(f"\n─ {os.path.basename(p)} ─")
        base = os.path.splitext(os.path.basename(p))[0]

        segmap, rgb, depth, cam_K, pc_full, pc_colors = \
            load_available_input_data(p, K=K)

        if segmap is None and (local_regions or filter_grasps):
            raise ValueError("Segmentation required for local region / filtering.")

        # ─ convert depth to cloud if needed ─
        if pc_full is None:
            if depth is None or cam_K is None:
                print("Skipping – no depth or intrinsics.")
                continue
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb, segmap_id=segmap_id,
                skip_border_objects=skip_border_objects, z_range=z_range)
        else:
            mask = (z_range[0] < pc_full[:,2]) & (pc_full[:,2] < z_range[1])
            pc_full   = pc_full[mask]
            if pc_colors is not None:
                pc_colors = pc_colors[mask]
            if segmap is not None:
                segmap = segmap[mask] if segmap.ndim == 1 else segmap

            pc_segments = {}
            if segmap is not None:
                ids = [segmap_id] if segmap_id is not None else np.unique(segmap[segmap>0])
                for sid in ids:
                    m = (segmap == sid)
                    if m.any(): pc_segments[sid] = pc_full[m]

        if pc_full is None or pc_full.shape[0] == 0:
            print("Skipping – empty cloud after filtering.")
            continue

        # ─ grasp prediction ─
        g_cam, scores, contacts, openings = grasp_estimator.predict_scene_grasps(
            sess, pc_full,
            pc_segments=pc_segments if pc_segments else None,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes)

        # wrap to dict if method returned arrays
        if not isinstance(g_cam, dict):
            g_cam  = {0: g_cam}
            scores = {0: scores}
            openings = {0: openings} if openings is not None else None

        # save raw results
        np.savez(f"results/npz_predictions/predictions_{base}.npz",
                 pred_grasps_cam=g_cam,
                 scores=scores,
                 contact_pts=contacts,
                 gripper_openings=openings)
        print("NPZ saved.")

        # ─ 3-D visualisation ─
        img_path = None
        if save_visualization_dir:
            img_path = os.path.join(save_visualization_dir, f"3d_{base}.png")
        visualize_grasps_o3d(
            pc_full,
            g_cam,
            scores,
            plot_opencv_cam=True,
            pc_colors=pc_colors,
            gripper_openings=openings,
            save_path=img_path)

    if not glob.glob(input_paths):
        print("No matching files for pattern:", input_paths)

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir",  default="/contact_graspnet/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001")
    ap.add_argument("--np_path",   default="test_data/*.npy")
    ap.add_argument("--png_path",  default="")
    ap.add_argument("--K",         default=None)
    ap.add_argument("--z_range",   default=[0.2,1.8],     type=lambda s: eval(str(s)))
    ap.add_argument("--local_regions",        action="store_true")
    ap.add_argument("--filter_grasps",        action="store_true")
    ap.add_argument("--skip_border_objects",  action="store_true")
    ap.add_argument("--forward_passes", type=int, default=1)
    ap.add_argument("--segmap_id",    type=int,  default=None)
    ap.add_argument("--arg_configs",  nargs="*", default=[])
    ap.add_argument("--save_visualization_dir")
    FLAGS = ap.parse_args()

    config = config_utils.load_config(
        FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    inference(
        config,
        FLAGS.ckpt_dir,
        FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path,
        K=FLAGS.K,
        z_range=FLAGS.z_range,
        local_regions=FLAGS.local_regions,
        filter_grasps=FLAGS.filter_grasps,
        segmap_id=FLAGS.segmap_id,
        forward_passes=FLAGS.forward_passes,
        skip_border_objects=FLAGS.skip_border_objects,
        save_visualization_dir=FLAGS.save_visualization_dir,
    )
