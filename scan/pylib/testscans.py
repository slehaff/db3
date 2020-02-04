import numpy as np
import cv2
import argparse
import sys
import os
from PIL import Image
import os
from makewrap import *
from unwrap import *
from jsoncloud import *

high_freq = 15
low_freq = .7
rwidth = 400
rheight = 400

def unwrap2():
    folder = '/home/samir/db3/scan/static/scan_folder/scan_im_folder/'
    # ref_folder = '/home/samir/db3/scan/static/scan_folder/scan_ref_folder'
    unwrap_r('lf_wrap.npy', 'hf_wrap.npy', folder )    # deduct_ref('unwrap.npy', 'unwrap.npy', folder, ref_folder)
    # generate_json_pointcloud(folder + 'image2.png', folder + 'unwrap.png', folder +'pointcl.json')
    generate_pointcloud(folder + 'image2.png', folder + 'unwrap.png', folder +'pointcl.ply')
    
    return

unwrap2()