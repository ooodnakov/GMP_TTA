import open3d as o3d
import numpy as np
import os, sys, time, yaml
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
from scipy import stats

from collections import deque
from numpy.linalg import inv
import pickle

from utils.one import *
from utils.two import *
import logging
root = logging.getLogger()
root.setLevel(logging.INFO)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# log_format = '%(name)s - %(levelname)s - %(message)s'
# formatter = logging.Formatter(log_format)
# handler.setFormatter(formatter)
# root.addHandler(handler)
log =logging.getLogger(__name__)

def load_kitti_scan(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    return scan.reshape((-1, 4))

def transform_cloud_to_world_frame(xyz, pose):
    points = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)))
    transformed_points = pose @ points.T

    return transformed_points.T[:,:3]

def build_kitti_map(base_path,second_path, start=0, end=None):

    # Load KITTI poses
    poses = parse_poses(os.path.join(second_path,"poses.txt"), parse_calibration(os.path.join(second_path,"calib.txt")))
    universal_scan = SemLaserScan(color_dict)
    all_points = []
    all_remissions = []
    all_labels = []
    for idx, pose in enumerate(tqdm(poses[start:end])):
        scan_filename = os.path.join(base_path, "velodyne", f'{idx:06d}.bin')
        label_filename = os.path.join(base_path, "labels", f'{idx:06d}.label')
        
        if not os.path.isfile(scan_filename):
            continue
        universal_scan.open_scan(scan_filename)
        universal_scan.open_label(label_filename, colorize=False)
        
        transformed_scan = transform_cloud_to_world_frame(universal_scan.points, pose)
    
        new_points = transform_cloud_to_world_frame(universal_scan.points, pose)
        new_remissions = universal_scan.remissions
        new_labels = universal_scan.sem_label
        all_points.append(new_points)
        all_remissions.append(new_remissions)
        all_labels.append(new_labels)
        
    all_points = np.vstack(all_points)
    all_remissions = np.hstack(all_remissions)
    all_labels = np.hstack(all_labels)
    universal_scan.set_points(all_points, all_remissions)
    universal_scan.set_label(all_labels)

    return universal_scan

# set your KITTI point cloud directory and poses file
base_path = "/home/alex_odnakov/datasets/kitti_dataset/sequences/"+sys.argv[1]
calib_path = "/home/alex_odnakov/personal/dataset/sequences/"+sys.argv[1]

# build the global map
log.info('starting builing')
global_cloud = build_kitti_map(base_path,calib_path,int(sys.argv[2]), int(sys.argv[3]))

ds_size_index=4
if len(sys.argv)>4:
    log.info('starting downsamplimng')
    sample_rate = float(sys.argv[ds_size_index])
    global_cloud.ds(sample_rate)

if len(sys.argv)>4:
    log.info('starting saving '+f'temp{sys.argv[ds_size_index]}.pkl')
    with open(f'temp{sys.argv[ds_size_index]}.pkl', 'wb') as file: 
        pickle.dump(global_cloud, file)
else:
    log.info('starting saving')
    log.info('%s, %s',global_cloud.voxel_size, global_cloud.ds_points is None)
    global_cloud.save(calib_path, f'whole_map_{int(sys.argv[2])}_{int(sys.argv[3])}')
    log.info('finishing saving')
    # log.info('starting colorzing')
    # global_cloud.colorize()
    # log.info('finished colorzing')
    # log.info('starting showing')
    # global_cloud.show()
    # log.info('finished showing')
    