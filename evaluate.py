import os
import torch
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from graph import Graph
from model import GNet
from pool import FeaturePooling
from metrics import chamfer_loss, loss_function, f1_score
from data import CustomDatasetFolder

# Args
parser = argparse.ArgumentParser(description='Pixel2Mesh evaluating script')
parser.add_argument('--data', type=str, metavar='D',
                    help="folder where data is located.")
parser.add_argument('--log_step', type=int, default=100, metavar='L',
                    help='how many batches to wait before logging evaluation status')
parser.add_argument('--output', type=str, default=None, metavar='G',
                    help='if not None, generate meshes to this folder')
parser.add_argument('--show_img', type=bool, default=False, metavar='S',
                    help='whether or not to show the images')
parser.add_argument('--load', type=str, metavar='M',
                    help='model file to load for evaluating.')
args = parser.parse_args()

# Model
model_gcn = GNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(args.load, map_location=device)
model_gcn.load_state_dict(state_dict)

# Turn batch norm into eval mode
for child in model_gcn.feat_extr.children():
    for ii in range(len(child)):
        if type(child[ii]) == torch.nn.BatchNorm2d:
            child[ii].track_running_stats = False

# Cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_gcn.cuda()
    print('Using GPU')
else:
    print('Using CPU')

# Graph
graph = Graph("./ellipsoid/init_info.pickle")

# Data Loader
folder = CustomDatasetFolder(args.data, extensions = ["dat"], print_ref=False)
val_loader = torch.utils.data.DataLoader(folder, batch_size=1, shuffle=True)

tot_loss_norm = 0
tot_loss_unorm = 0
tot_f1_1 = 0
tot_f1_2 = 0
tau = 1e-4
log_step = args.log_step
show_img = args.show_img

for n, data in enumerate(val_loader):
    im, gt_points, gt_normals = data

    if use_cuda:
        im = im.cuda()
        gt_points = gt_points.cuda()
        gt_normals = gt_normals.cuda()

    # Show image
    if show_img:
        img = im[0].float().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1))
        img = ((img * std + mean).transpose(1, 2, 0) * 255.0).round().astype(int)
        plt.imshow(img)
        plt.show()

    # Forward
    graph.reset()
    pool = FeaturePooling(im)
    pred_points = model_gcn(graph, pool)

    # Compute eval metrics
    _, loss_norm = chamfer_loss(pred_points[-1], gt_points.squeeze(), normalized=True)
    _, loss_unorm = chamfer_loss(pred_points[-1], gt_points.squeeze(), normalized=False)
    tot_loss_norm += loss_norm.item()
    tot_loss_unorm += loss_unorm.item()
    tot_f1_1 += f1_score(pred_points[-1], gt_points.squeeze(), threshold=tau)
    tot_f1_2 += f1_score(pred_points[-1], gt_points.squeeze(), threshold=2*tau)

    # Logs
    if n%log_step == 0:
        print("Batch", n)
        print("Normalized Chamfer loss so far", tot_loss_norm/(n+1))
        print("Unormalized Chamfer loss so far", tot_loss_unorm/(n+1))
        print("F1 score (tau=1e-4)", tot_f1_1/(n+1))
        print("F1 score (tau=2e-4)", tot_f1_2/(n+1))

    # Generate meshes
    if args.output is not None:
        graph.vertices = pred_points[5]
        graph.faces = graph.info[3][2]
        graph.to_obj(args.output + "plane_pred_block3_"
                        + str(n) + "_" + str(loss_norm.item()) + ".obj")
        graph.vertices = gt_points[0, :, :]
        graph.faces = []
        graph.to_obj(args.output + "plane_gt"
                        + str(n) + "_" + str(loss_norm.item()) + ".obj")
        print("Mesh plane_pred" + str(n) + "_" + str(loss_norm.item()) + " generated")

print("Final Normalized Chamfer loss:", tot_loss_norm/(n+1))
print("Final Unormalized Chamfer loss:", tot_loss_norm/(n+1))
print("Final F1 score (tau=1e-4) :", tot_f1_1/(n+1))
print("Final F1 score (tau=2e-4) :", tot_f1_2/(n+1))
