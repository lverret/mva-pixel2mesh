import os
import pickle
import argparse
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from graph import Graph
from model import GNet
from pool import FeaturePooling
from metrics import chamfer_loss

# Args
parser = argparse.ArgumentParser(description='Pixel2Mesh demoing script')
parser.add_argument('--data', type=str, metavar='D',
                    help="folder where images are located.")
parser.add_argument('--show_img', type=bool, default=False, metavar='S',
                    help='whether or not to show the images')
parser.add_argument('--load', type=str, metavar='M',
                    help='model file to load for demoing.')
parser.add_argument('--output', type=str, metavar='M',
                    help='folder where to save the generated meshes')
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

# Transform
demo_transforms = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])])

# Data Loader
demo_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(args.data, transform=demo_transforms),
                batch_size=1)

# Demoing
for n, (im, _) in enumerate(demo_loader):
    if use_cuda:
        im, _ = im.cuda()

    if args.show_img:
        img = im[0].float().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1))
        img = ((img * std + mean).transpose(1, 2, 0) * 255.0).round().astype(int)
        plt.imshow(img)
        plt.show()

    graph.reset()
    pool = FeaturePooling(im)
    pred_points = model_gcn(graph, pool)

    graph.vertices = pred_points[1]
    graph.faces = graph.info[1][2]
    graph.to_obj(args.output + "plane_pred_block1_" + str(n+1) + ".obj")
    graph.vertices = pred_points[3]
    graph.faces = graph.info[2][2]
    graph.to_obj(args.output + "plane_pred_block2_" + str(n+1) + ".obj")
    graph.vertices = pred_points[5]
    graph.faces = graph.info[3][2]
    graph.to_obj(args.output + "plane_pred_block3_" + str(n+1) + ".obj")
    print("Mesh %i generated!" %(n+1))
