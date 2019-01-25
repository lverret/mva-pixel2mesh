import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class VGG16(nn.Module):
    '''
    Pretrained VGG for image feature extraction
    '''
    def __init__(self):
        super(VGG16, self).__init__()
        model_conv = torchvision.models.vgg16(pretrained=True).features

        # Extract VGG feature maps conv_3, conv_4, conv_5
        layers = list(model_conv.children())
        self.conv3 = nn.Sequential(*layers[:-14])
        self.conv4 = nn.Sequential(*layers[:-7])
        self.conv5 = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv3(x), self.conv4(x), self.conv5(x)


class Block(nn.Module):
    '''
    Implement a mesh deformation block
    '''
    def __init__(self, feat_shape_dim):
        super(Block, self).__init__()
        self.conv1 = GCNConv(1280 + feat_shape_dim, 1024)
        self.conv21 = GCNConv(1024, 512)
        self.conv22 = GCNConv(512, 256)
        self.conv23 = GCNConv(256, 128)
        self.conv2 = [self.conv21, self.conv22, self.conv23]
        self.conv3 = GCNConv(128, 3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        '''
        Return 3D shape features (return[0]) and predicted 3D coordinates
        of the vertices (return[1])
        '''
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        for i in range(len(self.conv2)):
            conv = self.conv2[i]
            out = conv(out, edge_index)
            out = self.relu(out)
        return out, self.conv3(out, edge_index)

class GNet(torch.nn.Module):
    '''
    Implement the full cascaded mesh deformation network
    '''
    def __init__(self):
        super(GNet, self).__init__()
        self.feat_extr = VGG16()
        self.layer1 = Block(3) # No shape features for block 1
        self.layer2 = Block(128)
        self.layer3 = Block(128)

    def forward(self, graph, pool):
        # Initial ellipsoid mesh
        elli_points = graph.vertices.clone()

        # Layer 1
        features = pool(elli_points, self.feat_extr)
        input = torch.cat((features, elli_points), dim=1)
        x, coord_1 = self.layer1(input, graph.adjacency_mat[0])
        graph.vertices = coord_1

        # Unpool graph
        x = graph.unpool(x)
        coord_1_1 = graph.vertices.clone()

        # Layer 2
        features = pool(graph.vertices, self.feat_extr)
        input = torch.cat((features, x), dim=1)
        x, coord_2 = self.layer2(input, graph.adjacency_mat[1])
        graph.vertices = coord_2

        # Unpool graph
        x = graph.unpool(x)
        coord_2_1 = graph.vertices.clone()

        # Layer 3
        features = pool(graph.vertices, self.feat_extr)
        input = torch.cat((features, x), dim=1)
        x, coord_3 = self.layer3(input, graph.adjacency_mat[2])
        graph.vertices = coord_3

        return elli_points, coord_1, coord_1_1, coord_2, coord_2_1, coord_3

    def get_nb_trainable_params(self):
        '''
        Return the number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])
