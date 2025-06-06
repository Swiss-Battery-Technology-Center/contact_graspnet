import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- ADJUST THESE CONSTANTS FOR VISUAL APPEARANCE ---
GRIPPER_VISUAL_FORWARD_SHIFT = 0.06  # meters (6cm forward) - KEEPING THIS AGGRESSIVE

# Length of the main "finger" part extending forward
GRIPPER_FINGER_LENGTH_VIS = 0.08  # meters

# Length of the "approach tail" extending backward
GRIPPER_APPROACH_TAIL_LENGTH_VIS = 0.08 # meters

GRIPPER_LINE_RADIUS = 0.003  # Slightly thicker lines
GRIPPER_JOINT_RADIUS = 0.0045 # Slightly larger joints

# Color for the "finger base" lines (representing width)
FINGER_BASE_LINE_COLOR = [1.0, 0.6, 0.0] # Orange
# --- END ADJUSTABLE CONSTANTS ---


# Helper function to create a cylinder mesh (same as before)
def create_cylinder_segment(p1, p2, radius, color):
    p1 = np.asarray(p1); p2 = np.asarray(p2)
    vector = p2 - p1; length = np.linalg.norm(vector)
    if length < 1e-5: return None
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8, split=1)
    cylinder.paint_uniform_color(color)
    z_axis_target = vector / length
    x_axis_temp = np.array([1.0,0,0]) if np.abs(np.dot(z_axis_target,np.array([1.,0,0])))<0.95 else np.array([0,1.,0])
    y_axis_target = np.cross(z_axis_target, x_axis_temp); y_axis_target /= np.linalg.norm(y_axis_target)
    x_axis_target = np.cross(y_axis_target, z_axis_target); x_axis_target /= np.linalg.norm(x_axis_target)
    rotation_matrix = np.eye(3); rotation_matrix[:,0]=x_axis_target; rotation_matrix[:,1]=y_axis_target; rotation_matrix[:,2]=z_axis_target
    cylinder.rotate(rotation_matrix, center=(0,0,0)); cylinder.translate(p1, relative=False)
    return cylinder

# ────────────────────────────────────────────────────────────────
# 3-D grasp visualisation (Open3D) – "Pitchfork" Style
# ────────────────────────────────────────────────────────────────
def visualize_grasps_o3d(
    full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None,
    gripper_openings=None, gripper_width=0.08, save_path=None
):
    print("Preparing Open3D visualisation…")
    vis_geometries = []
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(full_pc)
    if pc_colors is not None:
        col = pc_colors.astype(np.float64);
        if col.max()>1.0: col/=255.0
        pcd.colors=o3d.utility.Vector3dVector(col)
    vis_geometries.append(pcd)
    cmap_obj=plt.get_cmap('tab10'); cmap_scores=plt.get_cmap('viridis')
    grasps_flat,scores_flat,openings_flat,colors_for_grasps_flat = [],[],[],[]
    # ... (Flattening logic - kept concise, assuming it's correct from before) ...
    if isinstance(pred_grasps_cam, dict):
        obj_idx = 0
        for obj_id, obj_grasps_list in pred_grasps_cam.items():
            if obj_grasps_list is None or len(obj_grasps_list) == 0: continue
            num_g = len(obj_grasps_list); base_c = cmap_obj(obj_idx % cmap_obj.N)[:3]
            s_list = scores.get(obj_id, [0.5]*num_g); o_list = gripper_openings.get(obj_id, [gripper_width]*num_g) if gripper_openings else [gripper_width]*num_g
            if len(s_list)!=num_g: s_list=[0.5]*num_g
            if len(o_list)!=num_g: o_list=[gripper_width]*num_g
            grasps_flat.extend(obj_grasps_list); scores_flat.extend(s_list); openings_flat.extend(o_list)
            colors_for_grasps_flat.extend([base_c]*num_g); obj_idx+=1
    else:
        if pred_grasps_cam is None or len(pred_grasps_cam) == 0: print("Warning: pred_grasps_cam (non-dict) is None or empty.")
        else:
            num_g = len(pred_grasps_cam); grasps_flat = list(pred_grasps_cam)
            s_flat = list(scores) if scores is not None and len(scores)==num_g else [0.5]*num_g
            o_flat = list(gripper_openings) if gripper_openings is not None and len(gripper_openings)==num_g else [gripper_width]*num_g
            scores_flat = s_flat; openings_flat = o_flat
            for s_val in scores_flat: colors_for_grasps_flat.append(cmap_scores(np.clip(float(s_val),0,1))[:3])

    MAX_GRASPS_TO_DRAW = 75 # Can increase if performance allows
    num_grasps_to_draw_actual = min(len(grasps_flat), MAX_GRASPS_TO_DRAW)
    
    skipped_due_to_width_count = 0

    for i in range(num_grasps_to_draw_actual):
        g_matrix = grasps_flat[i]
        current_width = openings_flat[i]
        current_grasp_color = colors_for_grasps_flat[i] # Main color for approach/finger length

        if not isinstance(g_matrix, np.ndarray) or g_matrix.shape != (4, 4): continue
        
        # Uncomment to debug widths:
        # print(f"Grasp {i}: width = {current_width:.4f}") 

        # If width is extremely small, the "base" lines will be points.
        # We can still draw the approach and the two main finger lines from the center.
        # Let's set a minimum visual width for the base lines if current_width is too small,
        # but still use current_width for point definition.
        # If current_width is truly < 1mm, skip to avoid issues.
        if current_width < 0.0005: # 0.5mm threshold
            skipped_due_to_width_count +=1
            continue

        # P0: Visual Palm Center (where approach ends, and finger bases start)
        p0_palm_center_local = np.array([0, 0, GRIPPER_VISUAL_FORWARD_SHIFT, 1])
        # P1: Visual Approach Tail
        p1_approach_tail_local = np.array([0, 0, GRIPPER_VISUAL_FORWARD_SHIFT - GRIPPER_APPROACH_TAIL_LENGTH_VIS, 1])

        # P2: End of Finger 1 Base Line (extends along +X from P0)
        p2_f1_base_end_local = np.array([current_width / 2, 0, GRIPPER_VISUAL_FORWARD_SHIFT, 1])
        # P3: Tip of Finger 1 Length Line (extends along +Z from P2)
        p3_f1_tip_local = np.array([current_width / 2, 0, GRIPPER_VISUAL_FORWARD_SHIFT + GRIPPER_FINGER_LENGTH_VIS, 1])

        # P4: End of Finger 2 Base Line (extends along -X from P0)
        p4_f2_base_end_local = np.array([-current_width / 2, 0, GRIPPER_VISUAL_FORWARD_SHIFT, 1])
        # P5: Tip of Finger 2 Length Line (extends along +Z from P4)
        p5_f2_tip_local = np.array([-current_width / 2, 0, GRIPPER_VISUAL_FORWARD_SHIFT + GRIPPER_FINGER_LENGTH_VIS, 1])

        # Transform all points to camera frame
        points_local = [p0_palm_center_local, p1_approach_tail_local, 
                        p2_f1_base_end_local, p3_f1_tip_local,
                        p4_f2_base_end_local, p5_f2_tip_local]
        
        points_cam = [(g_matrix @ p_loc)[:3] for p_loc in points_local]
        
        p0_palm_cam, p1_app_tail_cam, p2_f1_base_cam, p3_f1_tip_cam, p4_f2_base_cam, p5_f2_tip_cam = points_cam

        # Create Segments
        # 1. Approach Line
        cyl_approach = create_cylinder_segment(p1_app_tail_cam, p0_palm_cam, GRIPPER_LINE_RADIUS, current_grasp_color)
        if cyl_approach: vis_geometries.append(cyl_approach)

        # 2. Finger 1 Base Line
        cyl_f1_base = create_cylinder_segment(p0_palm_cam, p2_f1_base_cam, GRIPPER_LINE_RADIUS, FINGER_BASE_LINE_COLOR)
        if cyl_f1_base: vis_geometries.append(cyl_f1_base)
        
        # 3. Finger 1 Length Line
        cyl_f1_len = create_cylinder_segment(p2_f1_base_cam, p3_f1_tip_cam, GRIPPER_LINE_RADIUS, current_grasp_color)
        if cyl_f1_len: vis_geometries.append(cyl_f1_len)

        # 4. Finger 2 Base Line
        cyl_f2_base = create_cylinder_segment(p0_palm_cam, p4_f2_base_cam, GRIPPER_LINE_RADIUS, FINGER_BASE_LINE_COLOR)
        if cyl_f2_base: vis_geometries.append(cyl_f2_base)

        # 5. Finger 2 Length Line
        cyl_f2_len = create_cylinder_segment(p4_f2_base_cam, p5_f2_tip_cam, GRIPPER_LINE_RADIUS, current_grasp_color)
        if cyl_f2_len: vis_geometries.append(cyl_f2_len)

        # Joint Spheres
        joint_points_for_spheres = [p0_palm_cam, p2_f1_base_cam, p4_f2_base_cam]
        for jp_cam in joint_points_for_spheres:
            if np.any(np.isnan(jp_cam)): continue
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=GRIPPER_JOINT_RADIUS)
            sphere.translate(jp_cam); sphere.paint_uniform_color(current_grasp_color) # Use main color for joints
            vis_geometries.append(sphere)
    
    if skipped_due_to_width_count > 0:
        print(f"INFO: Skipped {skipped_due_to_width_count} grasps due to very small opening width (<0.0005m).")

    if plot_opencv_cam: vis_geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    view_params = {"window_name":"GraspNet Grasps (Open3D)", "width":1280, "height":720, "front":[0,-0.2,-1], "up":[0,-1,0], "zoom":0.7}
    if pcd.has_points(): view_params["lookat"]=pcd.get_center()
    else: view_params["lookat"]=[0,0,0]
    if save_path:
        vis=o3d.visualization.Visualizer(); vis.create_window(visible=False, width=view_params["width"], height=view_params["height"])
        for geom in vis_geometries: vis.add_geometry(geom)
        ctr=vis.get_view_control(); ctr.set_lookat(view_params["lookat"]); ctr.set_front(view_params["front"]); ctr.set_up(view_params["up"]); ctr.set_zoom(view_params["zoom"])
        vis.poll_events(); vis.update_renderer(); vis.capture_screen_image(save_path,do_render=True); vis.destroy_window(); print(f"Screenshot saved → {save_path}")
    else: o3d.visualization.draw_geometries(vis_geometries, **view_params)
