'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''
import os
import shutil
import sys
from subprocess import call

import numpy as np
from PIL import Image

from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj
from models import load_model

if sys.version_info < (3, 0):
    raise Exception("Please follow the installation \
    instruction on 'https://github.com/chrischoy/3D-R2N2'")

DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'

pred_file_name = ''


def set_pred_file_name(name):
    global pred_file_name
    pred_file_name = name


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(
            ['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy', '--create-dirs', '-o', fn])


def load_demo_images():
    ims = []
    for i in range(3):
        im = Image.open('imgs/%d.png' % i)
        # im = Image.open('imgs/%d.jpg' % (i+3))
        im = im.resize((127, 127))
        ims.append([np.array(im).transpose((2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    global pred_file_name
    if not cfg.TEST.MULTITEST or pred_file_name == '':
        pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    # load images
    demo_imgs = load_demo_images()

    # Download and load pretrained weights
    download_model(DEFAULT_WEIGHTS)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network
    net.load(DEFAULT_WEIGHTS)  # load downloaded weights
    solver = Solver(net)  # instantiate a solver

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)
    # print(voxel_prediction[0, :, 1, :, :])

    # Save the prediction to an OBJ file (mesh file).
    # print(voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)
    # print(type(voxel_prediction[0, :, 1, :, :]))
    # print(type(cfg.TEST.VOXEL_THRESH))
    # print(voxel_prediction[0, :, 1, :, :].shape)
    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)  # 0.4

    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    # if cmd_exists('meshlab'):
    #     call(['meshlab', pred_file_name])
    # else:
    #     print('Meshlab not found: please use visualization of your choice to view %s' %
    #           pred_file_name)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
