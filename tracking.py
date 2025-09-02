import pyzed.sl as sl
import numpy as np
import cv2
import apriltag
import json
import sys
import os
import math

class SuppressOutput:
    """Context manager to suppress stderr."""
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr.close()
        sys.stderr = self._stderr

def progress_bar(percent_done, bar_length=50):
    """Display a progress bar."""
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('\r[%s] %i%s' % (bar, percent_done, '%'))
    sys.stdout.flush()

# Directory containing subfolders with SVO2 files
root_dir = "/media/username/DiskExFAT/RoboticsPhantom/"

# Find all SVO2 files
svo_files = []
for root, dirs, files in os.walk(root_dir):
    
    for file in files:
        if file.endswith(".svo2"):
            svo_files.append(os.path.join(root, file))

if not svo_files:
    print("[INFO] No valid SVO2 files found. Exiting.")
    sys.exit()

# AprilTag Detector
detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
valid_tag_ids = {0, 2, 3}

# Process each SVO2 file
for svo_path in svo_files:
    print(f"\n[INFO] Processing file: {svo_path}")

    # Initialize ZED Camera
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Failed to open {svo_path}. Skipping...")
        continue

    num_frames = zed.get_svo_number_of_frames()
    camera_info = zed.get_camera_information()
    camera_params = camera_info.camera_configuration.calibration_parameters

    fx, fy = camera_params.left_cam.fx, camera_params.left_cam.fy
    cx, cy = camera_params.left_cam.cx, camera_params.left_cam.cy
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)  # Assuming no distortion

    # Output directory and filenames
    output_dir = os.path.dirname(svo_path)
    base_filename = os.path.splitext(os.path.basename(svo_path))[0]  # Remove extension
    json_files = {tag_id: os.path.join(output_dir, f"apriltag_{tag_id}.json") for tag_id in valid_tag_ids}
    
    fps = 15
    last_timestamp = None
    last_frame = None
    expected_interval_sec = 1.0 / fps
    skipped_log_path = os.path.join(output_dir, f"skipped_frames_log_annotated.txt")
    log_skips = open(skipped_log_path, "w")
    log_skips.write("Skipped Frame Log\nFrameIndex,TimeGap(s)\n")

    # Video output
    video_output_path = os.path.join(output_dir, f"fixed_annotated_console_view.avi")
    frame_width = zed.get_camera_information().camera_configuration.resolution.width
    frame_height = zed.get_camera_information().camera_configuration.resolution.height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    frame_count = 0

    try:
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                progress_bar((frame_count / num_frames) * 100)

                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                frame = image.get_data()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Get timestamp in seconds
                curr_timestamp_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                curr_timestamp = curr_timestamp_ns / 1e9

                
                # If not the first frame, check for skip
                if last_timestamp is not None:
                    gap = curr_timestamp - last_timestamp
                    if gap > expected_interval_sec * 1.5:
                        log_skips.write(f"{frame_count},{gap:.6f}\n")
                        # print(f"[SKIP] Large gap at frame {frame_count}: {gap:.3f}s. Inserting duplicate.")
                        # Write the last frame to maintain timing
                        if last_frame is not None:
                            out.write(last_frame)

                # Update last timestamp
                last_timestamp = curr_timestamp


                with SuppressOutput():
                    detections = detector.detect(gray)

                image_width, image_height = image.get_width(), image.get_height()

                for det in detections:
                    tag_id = det.tag_id
                    if tag_id not in valid_tag_ids:
                        continue

                    corners = det.corners.astype(np.float32)
                    if any(c[0] < 0 or c[1] < 0 or c[0] >= image_width or c[1] >= image_height for c in corners):
                        continue
                    
                    tag_size = 0.03/(2.0 ** 0.5)
                    half_size = tag_size/2.0
                    obj_points = np.array([
                        [-half_size,  half_size, 0],
                        [ half_size,  half_size, 0],
                        [ half_size, -half_size, 0],
                        [-half_size, -half_size, 0]
                    ], dtype=np.float32)

                    ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
                    if not ret:
                        continue

                    cX, cY = int(np.mean(corners, axis=0)[0]), int(np.mean(corners, axis=0)[1])
                    tag_depth = depth.get_value(cX, cY)[1]

                    # skip invalid depth values
                    if not np.isfinite(tag_depth) or tag_depth <= 0:
                        continue   

                    R, _ = cv2.Rodrigues(rvec)
                    z_axis = np.cross(R[:, 0], R[:, 1])
                    R[:, 2] = z_axis

                    T = tvec.reshape((3, 1))
                    # T[2] = tag_depth

                    Trans = np.hstack([R, T])
                    Trans = np.vstack((Trans, [0, 0, 0, 1]))
                    Trans_list = Trans.tolist()

                    tag_info = {
                        "tag_id": tag_id,
                        "frame_id": frame_count,
                        "position": {
                            "x": float(tvec[0]),
                            "y": float(tvec[1]),
                            "z": float(tag_depth)
                        },
                        "transformation_matrix": Trans_list
                    }

                    with open(json_files[tag_id], "a") as f:
                        json.dump(tag_info, f, indent=4)
                        f.write("\n")

                    # ------------- Draw tag border -----------------

                    for i in range(4):
                        pt1 = tuple(corners[i].astype(int))
                        pt2 = tuple(corners[(i + 1) % 4].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # ----------- Draw center and text --------------
                    cX, cY = np.mean(det.corners, axis=0).astype(int)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {det.tag_id}", (cX - 10, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Depth: {tag_depth:.2f}m", (cX - 10, cY + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    cv2.drawFrameAxes(frame,camera_matrix,dist_coeffs, rvec, tvec, 0.04)


                cv2.imshow("AprilTag Detection", frame)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame
                out.write(frame_bgr)

                last_frame = frame_bgr.copy()  # Keep a copy to reuse if needed

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                progress_bar(100, 30)
                sys.stdout.write("\n[INFO] SVO end reached. Moving to next file...\n")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User terminated. Moving to next file...")
                break

    except Exception as e:
        print(f"\n[ERROR] Error processing {svo_path}: {e}")

    finally:
        if out.isOpened():
            out.release()
        cv2.destroyAllWindows()
        log_skips.close()
        zed.close()

    print(f"[INFO] Processing complete for {svo_path}. Data saved.")

print("\n[INFO] All files processed.")
