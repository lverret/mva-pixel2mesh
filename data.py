import torch
import pickle
import numpy as np
import os

class CustomDatasetFolder(torch.utils.data.Dataset):
    '''
    Data reader
    '''
    def __init__(self, root, extensions, print_ref=False):
        self.samples = self._make_dataset(root, extensions)
        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        # Normalization for VGG
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1))
        # Print 3D model _ref
        self.print_ref = print_ref

    def __getitem__(self, index):
        path = self.samples[index]
        im, points, normals = self._loader(path)

        # Apply small transform
        im = im.astype(float)/255.0
        im = np.transpose(im, (2, 0, 1))
        im = (im - self.mean)/self.std

        return torch.from_numpy(im).float(), \
               torch.from_numpy(points).float(), \
               torch.from_numpy(normals).float()

    def __len__(self):
        return len(self.samples)

    def _loader(self, path):
        if self.print_ref:
            print(path)
        f = open(path, 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        im, data = u.load()
        points = data[:, :3]
        normals = data[:, 3:]
        return im, points, normals

    def _make_dataset(self, dir, extensions):
        paths = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = path
                        paths.append(item)
        return paths

    def _has_file_allowed_extension(self, filename, extensions):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)
