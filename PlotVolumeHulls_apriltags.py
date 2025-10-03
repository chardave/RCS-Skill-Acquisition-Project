import os
import json
import ast
import matplotlib
#matplotlib.use('Agg')
from datetime import datetime
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# Path to task segmentation Excel file
task_excel_path = r"F:/StarPhantom/timings.xlsx"
task_df = pd.read_excel(task_excel_path).set_index("Run File")

"""
Function to load marker data from JSON files: 
This function returns a dictionary of marker positions with marker ID as key 
and list of (frame_id, position) tuples as value
"""
def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6


def compute_convex_hull_volume(points, hull):
    # https://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy
    dt = Delaunay(points[hull.vertices])
    tets = dt.points[dt.simplices]
    volume = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))
    return volume

def compute_convex_hull_3d(points1, points2):
    # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.spatial.ConvexHull.html

    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    # print properties
    volume1 = compute_convex_hull_volume(points1, hull1)
    volume2 = compute_convex_hull_volume(points2, hull2)

    volumes = {}
    volumes['2'] = volume1
    volumes['3'] = volume2
    return volumes


def load_marker_data(file_paths):
    import math

    def is_nan(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    marker_positions = {}

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                buffer = ""
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    buffer += line
                    if line.endswith('}'):
                        try:
                            entry = json.loads(buffer, parse_constant=lambda x: float('nan') if x == 'NaN' else x)
                            buffer = ""  # reset after successful parse

                            marker_id = str(entry.get("tag_id"))
                            frame_id = entry.get("frame_id")
                            pos = entry.get("position", {})

                            if not isinstance(pos, dict):
                                continue

                            x, y, z = pos.get("x"), pos.get("y"), pos.get("z")
                            if any(is_nan(v) for v in (x, y, z)):
                                continue

                            if marker_id not in marker_positions:
                                marker_positions[marker_id] = []
                            marker_positions[marker_id].append((frame_id, (x, y, z)))

                        except Exception as e:
                            print(f"[!] Skipping one entry in {file_path}: {e}")
                            buffer = ""  # flush buffer and move on
        except Exception as e:
            print(f"[!] Could not read file {file_path}: {e}")

    for marker_id in marker_positions:
        marker_positions[marker_id].sort(key=lambda x: x[0])
    return marker_positions



"""
Function to interpolate positions:
When the timestamps of marker 6 don't match, we interpolate the positions of one marker to match the 
timestamps of the other marker.
"""
def interpolate_positions(frames, positions, target_frames):
    """
    Align control marker positions with target frames using the most recent (previous) value.
    """
    frames = np.array(frames)
    positions = np.array(positions)
    target_frames = np.array(target_frames)

    # Sort for safety
    sort_idx = np.argsort(frames)
    frames = frames[sort_idx]
    positions = positions[sort_idx]

    # Forward-fill logic
    interpolated_positions = []
    j = 0
    for t in target_frames:
        while j + 1 < len(frames) and frames[j + 1] <= t:
            j += 1
        interpolated_positions.append(positions[j])

    return np.stack(interpolated_positions, axis=0)

# Function to align markers relative to marker 6
def align_with_origin(marker_positions, control_marker_id, target_marker_ids):
    control_positions = marker_positions.get(control_marker_id, [])
    if not control_positions:
        return {}

    control_frames, control_coords = zip(*control_positions)
    aligned_positions = {}

    for marker_id in target_marker_ids:
        target_positions = marker_positions.get(marker_id, [])
        if not target_positions:
            continue

        target_frames, _ = zip(*target_positions)
        interpolated_control_coords = interpolate_positions(control_frames, control_coords, target_frames)

        aligned_positions[marker_id] = [
            (frame, tuple(np.array(position) - interpolated_control_coords[i]))
            for i, (frame, position) in enumerate(target_positions)
        ]
        
    return aligned_positions

def save_volumes_to_file(volumes, filename, marp, total_movement_left, total_movement_right, results_text):
    with open(filename, 'w') as file:
        for marker_id, volume in volumes.items():
            file.write(f"Marker ID: {marker_id}, Volume of Point Cloud: {volume:.4f} m^2\n") 
        file.write(f"Mean Absolute Relative Phase (MARP): {marp:.4f} radians\n")
        file.write(f"Total Movement Left Hand: {total_movement_left:.4f} m\n")
        file.write(f"Total Movement Right Hand: {total_movement_right:.4f} m\n")
        file.write(results_text)

def remove_outliers_with_frames(marker_positions, axis=2, method='iqr', threshold=1.5):
    """
    Remove outliers from marker positions while keeping frames aligned.
    """
    if not marker_positions:
        return marker_positions  # Return as is if empty

    filtered_positions = {}
    # frames_to_skip = 1850

    for marker_id, data in marker_positions.items():
        frames, positions = zip(*data)  # Separate frames and positions
        positions = np.array(positions)  # Convert positions to NumPy array

        # frames = frames[frames_to_skip:]  # Skip the first few frames
        # positions = positions[frames_to_skip:]

        # Get outlier bounds
        z_values = positions[:, axis]
        if method == 'iqr':
            Q1, Q3 = np.percentile(z_values, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'std':
            mean_z = np.mean(z_values)
            std_z = np.std(z_values)
            lower_bound = mean_z - threshold * std_z
            upper_bound = mean_z + threshold * std_z
        else:
            raise ValueError("Invalid method. Choose 'iqr' or 'std'.")

        # Filter both frames and positions
        valid_indices = (z_values >= lower_bound) & (z_values <= upper_bound)
        filtered_positions[marker_id] = [(frames[i], tuple(positions[i])) for i in range(len(frames)) if valid_indices[i]]

    return filtered_positions

# Function to plot both markers with colorbars
def plot_markers_with_colorbars(marker_positions, marker_ids_to_plot, plot_filename):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 14,
        'figure.titlesize': 18
    })
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('3D Positions of Markers 2 (Left Hand) and 3 (Right Hand) Relative to Marker 0', loc='right')

    colormaps = ['cool', 'Wistia']  # Different colormaps for markers
    scatter_plots = []

    for i, marker_id in enumerate(marker_ids_to_plot):
        frames, positions = zip(*marker_positions[marker_id])
        positions = np.array(positions)
        
        # duration = max(frames) - min(frames)
        duration = max(frames)
        # Normalize time for color intensity
        frame_values = np.linspace(0, 1, len(frames))
        cmap = plt.get_cmap(colormaps[i])
        norm = Normalize(vmin=0, vmax=1)

        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=frame_values, cmap=cmap, norm=norm, label=f'Marker {marker_id}', s=50
        )
        scatter_plots.append((scatter, duration, marker_id))

        if len(positions) >= 3:  # Convex hull requires at least 3 points
            hull = ConvexHull(positions)
            hull_color = 'darkblue' if marker_id == '2' else 'r'
            for simplex in hull.simplices:
                simplex = np.append(simplex, simplex[0])
                ax.plot(positions[simplex, 0], positions[simplex, 1], positions[simplex, 2], color=hull_color, linewidth=1.5)

    # Add individual colorbars for each marker
    for i, (scatter, duration, marker_id) in enumerate(scatter_plots):
        cbar = fig.colorbar(
            scatter, ax=ax, pad =0, shrink = 0.5, location='left',
            label=f'Frame Progression for Marker {marker_id}'
            )
        
        cbar.set_ticks([0, 1])
        cbar.ax.set_yticklabels([
            '0',
            duration
        ])

    # Set the view angle
    ax.view_init(elev=65, azim=90)  # Adjust the elevation and azimuthal angle as needed

    # ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0))
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.show()

def plot_task_colored_point_cloud(tasks_points, output_path):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 14,
        'figure.titlesize': 18
    })

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Task-Coloured Marker Positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Color pairs for each task
    colors = {
        "Dissection": ("#1f77b4", "#aec7e8"),
        "Continuous Suturing": ("#2ca02c", "#98df8a"),
        "Interrupted Suturing": ("#d62728", "#ff9896")
    }

    for task_name, (points2, points3) in tasks_points.items():
        color_left, color_right = colors.get(task_name, ("#000000", "#666666"))

        if points2 is not None:
            ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color=color_left, label=f"{task_name} - Left", alpha=0.6, s=20)
        if points3 is not None:
            ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2], color=color_right, label=f"{task_name} - Right", alpha=0.6, s=20)

    ax.view_init(elev=65, azim=90) 
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def remove_outliers(points, axis=2, method='iqr', threshold=1.5):
    """
    Remove outliers along a specified axis (default Z-axis).
    Methods:
    - 'iqr': Removes points outside Q1 - threshold*IQR and Q3 + threshold*IQR.
    - 'std': Removes points beyond mean ± threshold*std.
    """
    z_values = points[:, axis]

    if method == 'iqr':
        Q1, Q3 = np.percentile(z_values, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
    elif method == 'std':
        mean_z = np.mean(z_values)
        std_z = np.std(z_values)
        lower_bound = mean_z - threshold * std_z
        upper_bound = mean_z + threshold * std_z
    
    else:
        raise ValueError("Invalid method. Choose 'iqr' or 'std'.")

    return points[(z_values >= lower_bound) & (z_values <= upper_bound)]

def time_to_frames(time_str, framerate=30):
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds * framerate

def remove_frames(marker_positions, time_ranges, framerate=30):
    frames_to_remove = set()
    for start_time, end_time in time_ranges:
        start_frame = time_to_frames(start_time, framerate)
        end_frame = time_to_frames(end_time, framerate)
        frames_to_remove.update(range(start_frame, end_frame + 1))

    filtered_positions = {}
    for marker_id, data in marker_positions.items():
        filtered_positions[marker_id] = [(frame, pos) for frame, pos in data if frame not in frames_to_remove]

    return filtered_positions

def get_valid_marker_data(marker_positions, marker_id):
    data = marker_positions.get(marker_id, [])
    valid_data = []

    for f, p in data:
        try:
            # Defensive conversion of each element to float
            arr = np.array([
                float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).replace('-', '', 1).isdigit()) else np.nan
                if x not in ['NaN', 'Infinity', '-Infinity', None, '', 'inf', '-inf']
                else np.nan
                for x in p
            ], dtype=np.float32)

            # Skip any position with non-finite values
            if np.all(np.isfinite(arr)):
                valid_data.append((int(f), arr))
            else:
                print(f"[!] Non-finite position at marker {marker_id}, frame {f}: {p}")
        except Exception as e:
            print(f"[!] Error in get_valid_marker_data for marker {marker_id}, frame {f}")
            print(f"    Problematic position: {p} (type: {type(p)})")
            print(f"    Exception: {e}")

    if not valid_data:
        return None, None

    frames, positions = zip(*valid_data)
    return np.array(frames, dtype=int), np.array(positions, dtype=np.float32)



from scipy.interpolate import interp1d

def interpolate_marker_to_common_frames(frames, positions, target_frames):
    frames = np.array(frames)
    positions = np.array(positions)

    # Remove NaNs from positions and corresponding frames
    valid = ~np.isnan(positions).any(axis=1)
    frames = frames[valid]
    positions = positions[valid]

    # Remove duplicate frame values (keep last occurrence)
    unique_frames, unique_indices = np.unique(frames, return_index=True)
    frames = frames[unique_indices]
    positions = positions[unique_indices]

    # Sort just in case
    sort_idx = np.argsort(frames)
    frames = frames[sort_idx]
    positions = positions[sort_idx]

    # Now interpolate per axis safely
    interp_positions = []
    for dim in range(positions.shape[1]):
        interp_func = interp1d(
            frames, positions[:, dim],
            kind='linear', bounds_error=False, fill_value="extrapolate", assume_sorted=True
        )
        interp_positions.append(interp_func(target_frames))

    return np.stack(interp_positions, axis=1)

def compute_velocity(positions, fps=15):
    dt = 1.0 / fps
    return np.gradient(positions, dt, axis=0)

def compute_total_movement(positions):
    diffs = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(step_distances)
    return total_distance


def compute_phase_angles(positions, velocities, axis=2):
    pos_1d = positions[:, axis]
    vel_1d = velocities[:, axis]
    phase_angles = np.arctan2(vel_1d, pos_1d)  # safer than atan(vel/pos)
    return phase_angles

def compute_marp(phase_left, phase_right):
    relative_phase = phase_right - phase_left
    # Wrap to [-π, π]
    relative_phase_wrapped = (relative_phase + np.pi) % (2 * np.pi) - np.pi
    marp = np.mean(np.abs(relative_phase_wrapped))
    return marp, np.abs(relative_phase_wrapped)


def plot_phase_and_relative(frames, phase_left, phase_right, relative_phase, title_suffix='', plot_filename=''):
    # Downsample: every 5th value
    step = 5
    frames_ds = frames[::step]
    phase_left_ds = np.degrees(phase_left)[::step]
    phase_right_ds = np.degrees(phase_right)[::step]
    relative_phase_ds = np.degrees(relative_phase)[::step]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(frames_ds, phase_left_ds, label="Left Hand Phase")
    plt.plot(frames_ds, phase_right_ds, label="Right Hand Phase")
    plt.ylabel("Phase (degrees)")
    plt.ylim(-40, 40)
    plt.title("Phase Angles" + title_suffix)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(frames_ds, relative_phase_ds, label="Relative Phase")
    plt.axhline(np.mean(np.abs(np.degrees(relative_phase))), color='r', linestyle='--', label="MARP")
    plt.xlabel("Frame")
    plt.ylabel("Relative Phase (degrees)")
    plt.ylim(0, 40)
    plt.title("Relative Phase" + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight')



def compute_convex_volume(points):
    from scipy.spatial import ConvexHull
    if len(points) < 4:
        return 0.0
    try:
        return ConvexHull(points).volume
    except:
        return 0.0

def format_task_results(task_name, duration_s, dist_left, dist_right, vol_left, vol_right, marp_deg):
    return (
        f"\n--- {task_name} ---\n"
        f"Duration: {duration_s:.2f} seconds\n"
        f"Distance (Left Hand): {dist_left:.4f} m\n"
        f"Distance (Right Hand): {dist_right:.4f} m\n"
        f"Volume (Left Hand): {vol_left:.5f} m^3\n"
        f"Volume (Right Hand): {vol_right:.5f} m^3\n"
        f"MARP: {marp_deg:.2f} degrees\n"
        )

"""
start processing the data 
"""

# Base directory
base_dir = "F:/StarPhantomSession3+Expert"

# Loop through all run folders inside base_dir
for run_name in os.listdir(base_dir):
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.isdir(run_dir):
        continue

    file_paths = [
        os.path.join(run_dir, "fixed_apriltag_0_filled.json"),
        os.path.join(run_dir, "fixed_apriltag_2_filled.json"),
        os.path.join(run_dir, "fixed_apriltag_3_filled.json")
    ]

    # if not all(os.path.exists(fp) for fp in file_paths):
    #     print(f"[!] Skipping {run_name}: Missing apriltag files")
    #     continue

    print(f"\n=== Processing {run_name} ===")


    try:
        marker_positions = load_marker_data(file_paths)

        # Remove frames based on provided time ranges
        # time_ranges = [("00:00","0:23"), ("0:49","0:59"), ("1:52","2:17"), ("2:35","5:40"), ("6:10","7:20"), ("8:45","8:54"), ("9:29","10:01")]
        
        # marker_positions = remove_frames(marker_positions, time_ranges)


        # aligned_positions = align_with_origin(marker_positions, '0', ['2', '3'])
        aligned_positions=marker_positions
        # filtered_positions = aligned_position


        frames2, pos2 = get_valid_marker_data(aligned_positions, '2')
        frames3, pos3 = get_valid_marker_data(aligned_positions, '3')

        marp = None

        output_dir = os.path.join(base_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)

        if frames2 is not None or frames3 is not None:
            # Get common frame range
            common_start = max(frames2[0], frames3[0])
            common_end = min(frames2[-1], frames3[-1])
            common_frames = np.arange(common_start, common_end + 1)
            # Interpolate to common frames
            interp_pos2 = interpolate_marker_to_common_frames(frames2, pos2, common_frames)
            interp_pos3 = interpolate_marker_to_common_frames(frames3, pos3, common_frames)

            # Remove any rows with NaNs after interpolation
            valid_mask = ~np.isnan(interp_pos2).any(axis=1) & ~np.isnan(interp_pos3).any(axis=1)
            interp_pos2 = interp_pos2[valid_mask]
            interp_pos3 = interp_pos3[valid_mask]
            valid_frames = common_frames[valid_mask]

            if len(common_frames) < 10:
                print(f"Not enough common frames for {run_name}. Skipping MARP calculation.")
                continue

            vel2 = compute_velocity(interp_pos2, fps = 15)
            vel3 = compute_velocity(interp_pos3, fps = 15)

            phase2 = compute_phase_angles(interp_pos2, vel2, axis = 2)
            phase3 = compute_phase_angles(interp_pos3, vel3, axis = 2)

            total_movement_left = compute_total_movement(interp_pos2)
            total_movement_right = compute_total_movement(interp_pos3)

            marp, relative_phase = compute_marp(phase2, phase3)

            print(f"Mean Absolute Relative Phase (MARP) for {run_name}: {np.degrees(marp):.2f} degrees")
            phaseplot_file = os.path.join(output_dir, 'relative_phase.png')
            plot_phase_and_relative(valid_frames, phase2, phase3, relative_phase, title_suffix=f' ({run_name})', plot_filename=phaseplot_file)
       
        filtered_positions = remove_outliers_with_frames(aligned_positions, axis=2, method='std', threshold=1.96)
        # filtered_positions = aligned_positions
        # Extract points for Marker 2 and Marker 3
        points2 = np.array([pos for _, pos in aligned_positions.get('2', [])])
        points3 = np.array([pos for _, pos in aligned_positions.get('3', [])])
        
        # Remove frames from points7 and points8 based on provided time ranges
        # points7 = remove_frames_from_points(points7, time_ranges)
        # points8 = remove_frames_from_points(points8, time_ranges)
        print("checkpoint 1")
        # Remove Z-axis outliers
        points2 = remove_outliers(points2, axis=2, method='std', threshold=1.96)
        points3 = remove_outliers(points3, axis=2, method='std', threshold=1.96)
        print("checkpoint 2")
        #if points7.size == 0 or points8.size == 0:
        #    print(f"Skipping {run_name} as one or both markers have no aligned positions.")
        #    continue
        # Compute convex hulls and volumes
        volumes = compute_convex_hull_3d(points2, points3)

        if run_name in task_df.index:
            task_segments = {
                "Dissection": ("Dissection start", "Dissection End"),
                "Continuous Suturing": ("Continuous start", "Continuous End"),
                "Interrupted Suturing": ("Interrupted Start", "Interrupted End")
            }

            results_text = "\n=== Per-Task Analysis ==="
            taskwise_points = {}
            for task, (start_col, end_col) in task_segments.items():
                try:
                    start_frame = int(task_df.loc[run_name, start_col])
                    end_frame = int(task_df.loc[run_name, end_col])
                except Exception as e:
                    print(f"[!] Could not parse frame range for task {task} in {run_name}: {e}")
                    continue

                mask = (valid_frames >= start_frame) & (valid_frames <= end_frame)
                if not mask.any():
                    results_text += f"\n[!] No valid frames for {task} in {run_name}"
                    continue
         
                pos2_task = interp_pos2[mask]
                pos3_task = interp_pos3[mask]
                phase_task = relative_phase[mask]
                taskwise_points[task] = (pos2_task, pos3_task)


                duration = len(pos2_task) / 15  # fps
              
                dL = compute_total_movement(pos2_task)
                dR = compute_total_movement(pos3_task)
               
                vL = compute_convex_volume(pos2_task)
                vR = compute_convex_volume(pos3_task)
                marp_task = np.mean(phase_task)

                # Save per-task relative phase plot
                rel_phase_file = os.path.join(output_dir, f"relative_phase_{task}.png")
                plot_phase_and_relative(
                    valid_frames[mask],
                    phase2[mask],
                    phase3[mask],
                    phase_task,
                    title_suffix=f' ({run_name} - {task})',
                    plot_filename=rel_phase_file
                )

                results_text += format_task_results(task, duration, dL, dR, vL, vR, np.degrees(marp_task))

            colored_plot_file = os.path.join(output_dir, 'taskwise_colored_plot.png')
            plot_task_colored_point_cloud(taskwise_points, colored_plot_file)

        # Save volumes to file
    
        volumes_file = os.path.join(output_dir, 'point_cloud_volumeshull.txt')
        save_volumes_to_file(volumes, volumes_file, marp, total_movement_left, total_movement_right, results_text)


        plot_file = os.path.join(output_dir, 'markers_plothull.png')
        plot_markers_with_colorbars(filtered_positions, marker_ids_to_plot=['2', '3'], plot_filename=plot_file)

        plt.close('all')

    except Exception as e:
        print(f"Error processing {run_name}: {e}")



print("Processing complete!")
