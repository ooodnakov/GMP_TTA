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

from utils.two import *
from utils.one import *
import logging    # first of all import the module

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
log_format = '%(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
root.addHandler(handler)
log =logging.getLogger(__name__)

dataset = '/home/alex_odnakov/personal/dataset'
sequences_dir = os.path.join(dataset, "sequences")
sequence_folders = [
    f for f in sorted(os.listdir(sequences_dir))
    if os.path.isdir(os.path.join(sequences_dir, f))
]
scan = SemLaserScan(color_dict, project=True)
seq_i = int(sys.argv[1])
name = 'whole_map_0.1_ds_0.5vs'
scan.open_scan(os.path.join(sequences_dir,sequence_folders[seq_i],name + '.bin'))
scan.open_label(os.path.join(sequences_dir,sequence_folders[seq_i],name + '.label'))
log.info(scan.points.shape)
scan.colorize()
log.info('finished colorizing')
scan.show()