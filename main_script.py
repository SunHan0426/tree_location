"""
Main scripts for tree trunk detection and ranging in orchards by Han Sun

Usage:
    $ python main_script.py
"""
import subprocess
import pyzed.sl as sl
import cv2
import os
import numpy as np
import time
import logging
import sys

identifier = 1
id_ZED = 1
id_predict = 1

with open('id.txt', 'w') as file:
    file.write(f'id={id_predict}')

def calculate_euclidean_distance(point_cloud_np):
    # Calculate three-dimensional Euclidean distance
    x = point_cloud_np[:, :, 0]
    y = point_cloud_np[:, :, 1]
    z = point_cloud_np[:, :, 2]
    euclidean_distances = np.round(np.sqrt(x**2 + y**2 + z**2), 4)
    return euclidean_distances

def image_capture():
    #  Use ZED2i to obtain img and associated information
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
    i = 0
    image_left = sl.Mat()
    image_right = sl.Mat()
    dep_npy = sl.Mat()
    dep_view = sl.Mat()
    point_cloud = sl.Mat()

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        print("OPEN CAMERA SUCCESS.")
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        img_left = image_left.get_data()

        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        img_right = image_right.get_data()

        zed.retrieve_measure(dep_npy, sl.MEASURE.DEPTH)
        dep_map = dep_npy.get_data()

        zed.retrieve_image(dep_view, sl.VIEW.DEPTH)
        dep_visual = dep_view.get_data()

        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
        point_cloud_np = point_cloud.get_data()
        print("point_cloud_np shape: ", point_cloud_np.shape)
        start_time1 = time.time()
        euclidean_distance = calculate_euclidean_distance(point_cloud_np)
        print(euclidean_distance)
        end_time1 = time.time()
        elapsed_time1 = end_time1 - start_time1
        print(f"calculate distance time：{elapsed_time1:.2f} seconds")
        print("euclidean_distance shape: ", euclidean_distance.shape)

        view_RGB_LEFT = img_left
        view_RGB_RIGHT = img_right
        view_Depth = dep_visual

        nan_mask = np.isnan(euclidean_distance)
        print("contain nan value?", np.any(nan_mask))

        euclidean_distance_vis = (euclidean_distance - np.nanmin(euclidean_distance)) / (np.nanmax(euclidean_distance) -
                                                                                    np.nanmin(euclidean_distance)) * 255.0
        euclidean_distance_vis = euclidean_distance_vis.astype("uint8")
        euclidean_distance_vis = cv2.applyColorMap(euclidean_distance_vis, cv2.COLORMAP_VIRIDIS)

        SavePath = os.path.join("./img", "RGB_LEFT_{:0>3d}.png".format(id_ZED))
        SavePath_1 = os.path.join("./img", "RGB_RIGHT_{:0>3d}.png".format(id_ZED))
        SavePath_2 = os.path.join("./deepmap", "Deep_{:0>3d}.png".format(id_ZED))
        SavePath_5 = os.path.join("./deepmap", "Distance_{:0>3d}.png".format(id_ZED))
        SavePath_3 = os.path.join("./npy", "deep_npy_ZED_{:0>3d}.npy".format(id_ZED))
        SavePath_4 = os.path.join("./npy", "distance_npy_ZED_{:0>3d}.npy".format(id_ZED))

        cv2.imwrite(SavePath, view_RGB_LEFT)
        cv2.imwrite(SavePath_1, view_RGB_RIGHT)
        print("{:0>3d} RGB IMAGE SAVED.".format(id_ZED))
        cv2.imwrite(SavePath_2, view_Depth)
        print("{:0>3d} DEEP IMAGE SAVED.".format(id_ZED))
        np.save(SavePath_3, dep_map)
        print("{:0>3d} DEEP npy SAVED.".format(id_ZED))
        np.save(SavePath_4, euclidean_distance)
        print("{:0>3d} DISTANCE npy SAVED.".format(id_ZED))
        cv2.imwrite(SavePath_5, euclidean_distance_vis)
        print("{:0>3d} DISTANCE IMAGE SAVED.".format(id_ZED))

    zed.close()

def run_predict():
    # Run improved PSPNet to detect trunk
    try:
        subprocess.run(["python", "./pspnet/predict.py"])
    except FileNotFoundError:
        print("NO SUCH SCRIPT.")

while True:

    log_file = f'logs/log_{identifier:0>3d}.txt'
    if not os.path.exists(log_file):
        with open(log_file, 'w'):
            pass

    logging.basicConfig(filename=log_file, level=logging.INFO)

    class LoggerWriter:
        def __init__(self):
            self.terminal = sys.stdout

        def write(self, message):
            logging.info(message)
            sys.__stdout__.write(message)

        def flush(self):
            pass

    sys.stdout = LoggerWriter()

    user_input = input("TYPE 'Q' TO RUN SCRIPT：")

    if user_input.upper() == 'Q':

        print("main_script: EPOCH: {:0>3d}". format(identifier))
        image_capture()

        print("TRUNK LOCATING...")
        time.sleep(1.5)

        run_predict()

        id_ZED += 1

        identifier += 1

        id_predict += 1
        with open('id.txt', 'w') as file:
            file.write(f'id={id_predict}')

        while True:
            user_input = input("TYPE 'W' TO CONTROL ROD/TYPE 'BACK' TO RETURN：")
            if user_input.upper() == 'W':
                print("THIS IS TEST CODE(ROD_CTRL).")

                while True:
                    user_input = input("TYPE 'E' TO TAKE LEAF IMAGE/TYPE 'BACK' TO RETURN：")
                    if user_input.upper() == 'E':
                        print("THIS IS TEST CODE(LEAF_IMAGE_CAPTURE).")
                    elif user_input.upper() == 'BACK':
                        print("RETURN TO SECOND LAYER.")
                        break
                    else:
                        print("UNKNOWN COMMAND...3")

            elif user_input.upper() == 'BACK':
                print("RETURN TO FIRST LAYER.")
                break
            else:
                print("UNKNOWN COMMAND...2")

    elif user_input.upper() == 'ESC':
        print("PROCESS SHUTDOWN")
        break
    else:
        print("UNKNOWN COMMAND...1")