import open3d as o3d
import numpy as np
import os, sys, time, yaml
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
from scipy import stats
import pandas as pd
from collections import deque
from numpy.linalg import inv
import pickle
from multiprocessing import Pool
import logging


root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
log_format = '%(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
root.addHandler(handler)
log =logging.getLogger(__name__)


CFG = yaml.safe_load(open('/home/alex_odnakov/personal/gits/semantic-kitti-api/config/semantic-kitti.yaml', 'r'))
color_dict = CFG["color_map"]
learning_map_inv = CFG["learning_map_inv"]
learning_map = CFG["learning_map"]
color_dict = {key:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

class LaserScan:
	"""Class that contains LaserScan with x,y,z,r"""
	EXTENSIONS_SCAN = ['.bin']

	def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
		self.project = project
		self.proj_H = H
		self.proj_W = W
		self.proj_fov_up = fov_up
		self.proj_fov_down = fov_down
		self.reset()

	def reset(self):
		""" Reset scan members. """
		self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
		self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

		# projected range image - [H,W] range (-1 is no data)
		self.proj_range = np.full((self.proj_H, self.proj_W), -1,
								dtype=np.float32)

		# unprojected range (list of depths for each point)
		self.unproj_range = np.zeros((0, 1), dtype=np.float32)

		# projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
		self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
								dtype=np.float32)

		# projected remission - [H,W] intensity (-1 is no data)
		self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
									dtype=np.float32)

		# projected index (for each pixel, what I am in the pointcloud)
		# [H,W] index (-1 is no data)
		self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
								dtype=np.int32)

		# for each point, where it is in the range image
		self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
		self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

		# mask containing for each pixel, if it contains a point or not
		self.proj_mask = np.zeros((self.proj_H, self.proj_W),
								dtype=np.int32)       # [H,W] mask

	def size(self):
		""" Return the size of the point cloud. """
		return self.points.shape[0]

	def __len__(self):
		return self.size()

	def open_scan(self, filename):
		""" Open raw scan and fill in attributes
		"""
		# reset just in case there was an open structure
		self.reset()

		# check filename is string
		if not isinstance(filename, str):
				raise TypeError("Filename should be string type, "
					  "but was {type}".format(type=str(type(filename))))

		# check extension is a laserscan
		if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
				raise RuntimeError("Filename extension is not valid scan file.")

		# if all goes well, open pointcloud
		scan = np.fromfile(filename, dtype=np.float32)
		scan = scan.reshape((-1, 4))

		# put in attribute
		points = scan[:, 0:3]    # get xyz
		remissions = scan[:, 3]  # get remission
		self.set_points(points, remissions)

	def set_points(self, points, remissions=None):
		""" Set scan attributes (instead of opening from file)
		"""
		# reset just in case there was an open structure
		self.reset()

		# check scan makes sense
		if not isinstance(points, np.ndarray):
			raise TypeError("Scan should be numpy array")

		# check remission makes sense
		if remissions is not None and not isinstance(remissions, np.ndarray):
			raise TypeError("Remissions should be numpy array")

		# put in attribute
		self.points = points    # get xyz
		if remissions is not None:
			self.remissions = remissions  # get remission
		else:
			self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

		# if projection is wanted, then do it and fill in the structure
		if self.project:
			self.do_range_projection()

	def do_range_projection(self):
		""" Project a pointcloud into a spherical projection image.projection.
			Function takes no arguments because it can be also called externally
			if the value of the constructor was not set (in case you change your
			mind about wanting the projection)
		"""
		# laser parameters
		fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
		fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
		fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

		# get depth of all points
		depth = np.linalg.norm(self.points, 2, axis=1)

		# get scan components
		scan_x = self.points[:, 0]
		scan_y = self.points[:, 1]
		scan_z = self.points[:, 2]

		# get angles of all points
		yaw = -np.arctan2(scan_y, scan_x)
		pitch = np.arcsin(scan_z / (depth + 1e-8))

		# get projections in image coords
		proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
		proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

		# scale to image size using angular resolution
		proj_x *= self.proj_W                              # in [0.0, W]
		proj_y *= self.proj_H                              # in [0.0, H]

		# round and clamp for use as index
		proj_x = np.floor(proj_x)
		proj_x = np.minimum(self.proj_W - 1, proj_x)
		proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
		self.proj_x = np.copy(proj_x)  # store a copy in orig order

		proj_y = np.floor(proj_y)
		proj_y = np.minimum(self.proj_H - 1, proj_y)
		proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
		self.proj_y = np.copy(proj_y)  # stope a copy in original order

		# copy of depth in original order
		self.unproj_range = np.copy(depth)

		# order in decreasing depth
		indices = np.arange(depth.shape[0])
		order = np.argsort(depth)[::-1]
		depth = depth[order]
		indices = indices[order]
		points = self.points[order]
		remission = self.remissions[order]
		proj_y = proj_y[order]
		proj_x = proj_x[order]

		# assing to images
		self.proj_range[proj_y, proj_x] = depth
		self.proj_xyz[proj_y, proj_x] = points
		self.proj_remission[proj_y, proj_x] = remission
		self.proj_idx[proj_y, proj_x] = indices
		self.proj_mask = (self.proj_idx > 0).astype(np.float32)


class SemLaserScan(LaserScan):
	"""Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
	EXTENSIONS_LABEL = ['.label']

	def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
		super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
		self.reset()

		# make semantic colors
		max_sem_key = 0
		for key, data in sem_color_dict.items():
			if key + 1 > max_sem_key:
				max_sem_key = key + 1
			self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
		for key, value in sem_color_dict.items():
			self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

		# make instance colors
		max_inst_id = 100000
		self.inst_color_lut = np.random.uniform(low=0.0,
												high=1.0,
												size=(max_inst_id, 3))
		# force zero to a gray-ish color
		self.inst_color_lut[0] = np.full((3), 0.1)
		self.ds_points = None
		self.ds_remissions = None
		self.ds_sem_label = None
		self.ds_inst_label = None
		self.voxel_size = None

	def reset(self):
		""" Reset scan members. """
		super(SemLaserScan, self).reset()

		# semantic labels
		self.sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
		self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

		# instance labels
		self.inst_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
		self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

		# projection color with semantic labels
		self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
								 	dtype=np.int32)              # [H,W]	label
		self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
								 	dtype=float)              # [H,W,3] color

		# projection color with instance labels
		self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
										dtype=np.int32)              # [H,W]	label
		self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
										dtype=float)              # [H,W,3] color

	def open_label(self, filename, colorize=True):
		""" Open raw scan and fill in attributes
		"""
		# check filename is string
		if not isinstance(filename, str):
			raise TypeError("Filename should be string type, "
					  "but was {type}".format(type=str(type(filename))))

		# check extension is a laserscan
		if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
			raise RuntimeError("Filename extension is not valid label file.")

		# if all goes well, open label
		label = np.fromfile(filename, dtype=np.uint32)
		label = label.reshape((-1))

		# set it
		self.set_label(label)
		if colorize:
			self.colorize()

	def set_label(self, label):
		""" Set points for label not from file but from np
		"""
		# check label makes sense
		if not isinstance(label, np.ndarray):
			raise TypeError("Label should be numpy array")

		# only fill in attribute if the right size
		if label.shape[0] == self.points.shape[0]:
			self.sem_label = label & 0xFFFF  # semantic label in lower half
			self.inst_label = label >> 16    # instance id in upper half
		else:
			log.info("Points shape: %s", self.points.shape)
			log.info("Label shape: %s", label.shape)
			raise ValueError("Scan and Label don't contain same number of points")

		# sanity check
		assert((self.sem_label + (self.inst_label << 16) == label).all())

		if self.project:
			self.do_label_projection()

	def colorize(self):
		""" Colorize pointcloud with the color of each semantic label
		"""
		self.sem_label_color = self.sem_color_lut[self.sem_label]
		self.sem_label_color = self.sem_label_color.reshape((-1, 3))

		self.inst_label_color = self.inst_color_lut[self.inst_label]
		self.inst_label_color = self.inst_label_color.reshape((-1, 3))

		if self.ds_sem_label is not None:
			self.ds_sem_label_color = self.sem_color_lut[self.ds_sem_label]
			self.ds_sem_label_color = self.ds_sem_label_color.reshape((-1, 3))

			self.ds_inst_label_color = self.inst_color_lut[self.ds_inst_label]
			self.ds_inst_label_color = self.ds_inst_label_color.reshape((-1, 3))

	def do_label_projection(self):
		# only map colors to labels that exist
		mask = self.proj_idx >= 0

		# semantics
		self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
		self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

		# instances
		self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
		self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]

	def show(self, voxel=False, voxel_size=0.5,sample_rate=None,back='gl'):
		pcd = o3d.geometry.PointCloud()
		if self.ds_points is not None:
			pcd.points = o3d.utility.Vector3dVector(self.ds_points)
			pcd.colors = o3d.utility.Vector3dVector(self.ds_sem_label_color)
		else:
			pcd.points = o3d.utility.Vector3dVector(self.points)
			pcd.colors = o3d.utility.Vector3dVector(self.sem_label_color)
		if self.voxel_size is None:
			self.voxel_size = voxel_size
			pcd = pcd.voxel_down_sample(self.voxel_size)
		if sample_rate is not None:
			pcd = pcd.random_down_sample(sample_rate)
		if voxel:
			voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
															voxel_size=voxel_size)
			if back == 'gl':
				o3d.visualization.draw_geometries([voxel_grid])
			elif back == 'pl':
				o3d.visualization.draw_plotly([voxel_grid])
		else:
			if back == 'gl':
				o3d.visualization.draw_geometries([pcd])
			elif back == 'pl':
				o3d.visualization.draw_plotly([pcd])
   
	def pcd(self,voxel_size=0.5):
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.points)
		pcd.colors = o3d.utility.Vector3dVector(self.sem_label_color)
		if voxel_size is not None:
			pcd = pcd.voxel_down_sample(voxel_size)
			voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
														voxel_size=voxel_size)
			return voxel_grid
		else:
			return pcd

	def ds(self, sample_rate):
		assert sample_rate>0 and sample_rate<=1, 'Sample rate out of [0, 1]'
		if sample_rate<1:
			indices = np.random.choice(np.arange(0,self.points.shape[0]),int(self.points.shape[0]*sample_rate),replace=False)
			self.points = self.points[indices]
			self.sem_label = self.sem_label[indices]
			self.remissions = self.remissions[indices]

	def _calculate_mode(self,arr):
		unique_values, counts = np.unique(arr, return_counts=True)
		max_count_index = np.argmax(counts)
		mode_value = unique_values[max_count_index]
		return mode_value

	def _downsample(self, idx):
		# if idx % 10000==9999:
		# 	print(f'Done {idx+1:,} out {self.unique_indices_len:,}',end='/r')
		indices = np.where(self.unique_inverse == idx)[0]
		downsampled_xyz = np.mean(self.points[indices], axis=0)
		downsampled_remissions = np.mean(self.remissions[indices])
		return downsampled_xyz, downsampled_remissions, indices

	def voxel_ds(self, voxel_size=None, num_processes=4):
		if voxel_size is not None:
			self.voxel_size = voxel_size

		xyz = self.points
		indices = np.floor(xyz / self.voxel_size).astype(np.int32)
		unique_indices, unique_inverse = np.unique(indices, return_inverse=True, axis=0)
		self.unique_inverse = unique_inverse
		self.unique_indices_len = unique_indices.shape[0]
		log.info('Starting pool')
		# with Pool(processes=num_processes) as pool:
		# 	results = pool.map(self._downsample, range(unique_indices.shape[0]))
		results = []
		for idx in tqdm(range(unique_indices.shape[0])):
			results.append(self._downsample(idx))
		# Collect results
		downsampled_xyz, downsampled_remissions, all_indices = zip(*results)
		self.ds_points = np.vstack(downsampled_xyz)
		self.ds_remissions = np.hstack(downsampled_remissions)

		# Calculate modes for all voxels in a vectorized manner
		all_labels = np.hstack([self.sem_label[idx] for idx in all_indices])
		modes = [self._calculate_mode(self.sem_label[idx]) for idx in all_indices]


		if len(modes)== self.ds_points.shape[0]:
			self.ds_sem_label = np.array(modes) & 0xFFFF  # semantic label in lower half
			self.ds_inst_label = np.array(modes) >> 16    # instance id in upper half

		self.colorize()


	def save(self, file_path, file_name, voxel_size=None):
		if self.voxel_size is None:
			self.voxel_size = voxel_size
   
		if self.ds_points is 		None	and self.voxel_size is not None:
			log.info('Starting downsampling')
			self.voxel_ds()
			log.info('Finished downsampling')
   
		if 	self.voxel_size is not None:
			export = np.hstack((self.ds_points,	self.ds_remissions.reshape((-1,1)))	).reshape(-1).astype(np.float32)
		else:
			export = np.hstack((self.points, 	self.remissions.reshape((-1,1)))	).reshape(-1).astype(np.float32)
   
		export.tofile(os.path.join(file_path,file_name+'.bin'))
		self.sem_label.astype(np.uint32).tofile(os.path.join(file_path,file_name+'.label'))
