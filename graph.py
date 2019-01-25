import numpy as np
import pickle
import torch
from itertools import permutations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Graph:

    def __init__(self, info_filename):
        '''
        Read the pickle info file
        Structure of the loaded file:
            [[vertices_b0],
            [edges_b0, map_b0, faces_b0, pool_indices_b0],
            [edges_b1, map_b1, faces_b1, pool_indices_b1],
            [edges_b2, map_b2, faces_b2]]
        '''
        info_file = open(info_filename,"rb")
        info = pickle.load(info_file)
        # [[v0], [e0, m0, f0, idx[0]], [e1, m1, f1, idx[1]], [e2, m2, f2]]

        self.vertices = torch.Tensor(info[0][0]).float().to(device)
        self.edges = info[1][0]
        self.faces = info[1][2]
        self.idx = info[1][3]
        self.block_id = 1
        self.info = info
        adj_matrices = []
        maps = []
        len_maps = []
        for block_id in range(1, len(info)):
            # Build vertex-vertex map (useful for laplacian loss)
            map = info[block_id][1]
            len_map = torch.Tensor([len(l) for l in map]).view(-1, 1).to(device)
            map = [torch.Tensor(l).to(device) for l in map]
            # Pad to make tensor as vertices don't have same nb of neighbors
            map = torch.nn.utils.rnn.pad_sequence(map, True, -1).long().to(device)
            maps.append(map)
            len_maps.append(len_map)

            # Adjacency mat
            edges = info[block_id][0]
            mat = list(zip(*edges))
            adj_matrices.append(torch.tensor([list(mat[0]), list(mat[1])],
                                             dtype=torch.long).to(device))
        self.adjacency_mat = adj_matrices
        self.maps = maps
        self.len_maps = len_maps

    def reset(self):
        '''
        Reset graph to initial ellipsoid
        '''
        self.vertices = torch.Tensor(self.info[0][0]).float().to(device)
        self.edges = self.info[1][0]
        self.faces = self.info[1][2]
        self.idx = self.info[1][3]
        self.block_id = 1

    def to_obj(self, obj_file):
        '''
        Export the current graph to obj file
        '''
        # Make a copy out of the torch comput graph
        vertices_copy = self.vertices.cpu().clone().detach().numpy()

        # Writing vertices
        file = open(obj_file,"w")
        for id_vertex in range(vertices_copy.shape[0]):
            xcoord = str(vertices_copy[id_vertex, 0]).replace('.', ',')
            ycoord = str(vertices_copy[id_vertex, 1]).replace('.', ',')
            zcoord = str(vertices_copy[id_vertex, 2]).replace('.', ',')
            file.write("v " + xcoord + " " + ycoord + " " + zcoord + "\n")

        # Writing faces
        for face in self.faces:
            face0 = str(face[0])
            face1 = str(face[1])
            face2 = str(face[2])
            file.write("f " + face0 + " " + face1 + " " + face2 + "\n")

    def unpool(self, x=None):
        '''
        Graph unpooling layer
        Args:
            x: Vertex features
        Return:
            x: Vertex features with augmented vertices
        '''
        if self.block_id >= 3:
            raise RuntimeError("Maximum unpooling level already achieved (b3)")

        # Increase number of vertices
        new_v = torch.sum(self.vertices[self.idx], dim=1)/2
        self.vertices = torch.cat((self.vertices, new_v), dim = 0)

        # Interpol features for the extra vertices
        if x is not None:
            new_feat = torch.sum(x[self.idx], dim=1)/2
            x = torch.cat((x, new_feat), dim = 0)

        # Adapt edges and faces
        self.edges = self.info[self.block_id+1][0]
        self.faces = self.info[self.block_id+1][2]

        # Update pooling idx
        if self.block_id < 2:
            self.idx = self.info[self.block_id+1][3]
        self.block_id += 1

        return x

if __name__ == "__main__":
    print("Testing Graph")
    graph = Graph("./ellipsoid/init_info.pickle")
    print("Vertices dim (block 1):", graph.vertices.size())
    graph.unpool()
    print("Vertices dim (block 2):", graph.vertices.size())
    graph.unpool()
    print("Vertices dim (block 3):", graph.vertices.size())
    graph.reset()
    print("Vertices dim (block 1):", graph.vertices.size())
