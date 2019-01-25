import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeaturePooling():

    def __init__(self, im):
        self.im = im

    def __call__(self, points, feat_extr):
        # Project
        x, y = self._project(points)
        # Compute interpolated features
        feat_conv3, feat_conv4, feat_conv5 = feat_extr(self.im)
        interp_feat = self._pool_features(x, y, [feat_conv3, feat_conv4, feat_conv5])
        return interp_feat

    def _project(self, points):
        '''
        Project the 3D points onto the image plane using camera intrinsics
        '''
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        focal = 248
        x = focal * torch.div(-Y, -Z) + 111.5
        y = focal * torch.div(X, -Z) + 111.5

        x = torch.clamp(x, 0, 223)
        y = torch.clamp(y, 0, 223)

        return x, y

    def _pool_features(self, x, y, feat_list):
        '''
        Pool the features from four nearby pixels using bilinear interpolation
        '''
        concat_features = torch.FloatTensor().to(device)

        for feat in feat_list:

            d = feat.shape[2] - 1 # range from 0 to d - 1 (e.g 0 to 6)

            x_ext = (x / 224) * d
            y_ext = (y / 224) * d
            x1 = torch.floor(x_ext).long()
            x2 = torch.ceil(x_ext).long()
            y1 = torch.floor(y_ext).long()
            y2 = torch.ceil(y_ext).long()

            # Pool the four nearby pixels features
            f_Q11 = feat[0, :, x1, y1]
            f_Q12 = feat[0, :, x1, y2]
            f_Q21 = feat[0, :, x2, y1]
            f_Q22 = feat[0, :, x2, y2]

            # Bilinear interpolation
            w1 = x2.float() - x_ext
            w2 = x_ext - x1.float()
            w3 = y2.float() - y_ext
            w4 = y_ext - y1.float()
            feat_bilinear = f_Q11 * w1 * w3 + f_Q21 * w2 * w3 + \
                            f_Q12 * w1 * w4 + f_Q22 * w2 * w4

            concat_features = torch.cat((concat_features, feat_bilinear))

        return concat_features.t()


if __name__ == "__main__":
    from graph import Graph
    import matplotlib.pyplot as plt
    print("Testing Feature Pooling")
    graph = Graph("./ellipsoid/init_info.pickle")
    pool = FeaturePooling(None)
    x, y = pool._project(graph.vertices)
    x = x.numpy(); y = y.numpy()
    img = np.zeros((224,224,3), np.uint8)
    img[np.round(x).astype(int), np.round(y).astype(int), 2] = 0
    img[np.round(x).astype(int), np.round(y).astype(int), 1] = 255
    plt.imshow(img)
    plt.show()
