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
# handler.setLevel(logging.INFO)
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
    
def transform_cloud_to_origin_frame(xyz, pose):
    points = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)))
    transformed_points = np.linalg.inv(pose) @ points.T
    return transformed_points.T[:,:3]

# set your KITTI point cloud directory and poses file
base_path = "/home/alex_odnakov/personal/dataset/sequences/"+sys.argv[1]

calibration = parse_calibration(os.path.join(base_path, "calib.txt"))
poses = parse_poses(os.path.join(base_path, "poses.txt"), calibration)
poses = np.hstack(poses)

xyz_name = os.path.join(base_path, f'whole_map_{int(sys.argv[2])}_{int(sys.argv[3])}.bin')
xyz_l_name = os.path.join(base_path, f'whole_map_{int(sys.argv[2])}_{int(sys.argv[3])}.label')
xyz_w_rem = load_kitti_scan(xyz_name)
labels = np.fromfile(xyz_l_name, dtype=np.uint32)
xyz = xyz_w_rem[:,:3]
remissions = xyz_w_rem[:,3]

vel_vers = sys.argv[4]

log.info(f'starting velodyne{vel_vers} creation')

for idx in tqdm(range(260,int(sys.argv[3])-(29 if sys.argv[3]=='4100' else 250))):
    scan_filename = os.path.join(base_path, f"velodyne{vel_vers}", f'{idx:06d}.bin')
    labels_filename = os.path.join(base_path, f"labels{vel_vers}", f'{idx:06d}.label')
    if os.path.exists(scan_filename) and os.path.exists(labels_filename):
        continue
    orig_scan_filename = os.path.join(base_path, "velodyne", f'{idx:06d}.bin')
    orig_label_filename = os.path.join(base_path, "labels", f'{idx:06d}.label')
    orig_scan = load_kitti_scan(orig_scan_filename)
    orig_labels = np.fromfile(orig_label_filename, dtype=np.uint32)
    new_points = transform_cloud_to_origin_frame(xyz, poses[:,idx * 4: (idx + 1) * 4])
    depth = np.linalg.norm(new_points[:,:3], 2, axis=1)
    ind_filt = (depth < 80) & (depth > 4)
    

    
    if vel_vers=='2':
        freqs = plt.hist(np.floor(np.linalg.norm(orig_scan[:,:3], 2, axis=1)),bins=np.arange(0,81,8))
        ind_filt2 = np.random.choice(np.where(ind_filt)[0],depth[ind_filt].shape[0]//30,replace=False, p=freqs[0][np.floor(depth[ind_filt]).astype('int')//8]/freqs[0][np.floor(depth[ind_filt]).astype('int')//8].sum())
    elif vel_vers=='3':
        ind_filt2 = np.random.choice(np.where(ind_filt)[0],depth[ind_filt].shape[0]//30,replace=False)
    
    if not os.path.exists(scan_filename):
        export = np.hstack([new_points[ind_filt2],remissions[ind_filt2].reshape((-1,1))])
        export = np.vstack([export,orig_scan]).reshape(-1).astype(np.float32)
        export.tofile(scan_filename)
        del export
        
    if not os.path.exists(labels_filename):
        export_labels = np.hstack([labels[ind_filt2],orig_labels])
        export_labels.tofile(labels_filename)
        del export_labels
        
    del new_points, ind_filt, ind_filt2, depth, orig_scan, orig_labels
    
    if vel_vers=='2':
        del freqs
    # break
log.info(f'finished velodyne{vel_vers} creation')