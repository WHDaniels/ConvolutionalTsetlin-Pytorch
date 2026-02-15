import numpy as np
from torch.utils.data import Dataset, DataLoader
from math import ceil
import torch

class TMDataset(Dataset):
    def __init__(self, data_root, data_length, rank, args, pos_grid_size, stride=1):
        super().__init__()

        images_path = f'{data_root}_images.npy'
        labels_path = f'{data_root}_labels.npy'

        self.images = np.load(images_path, mmap_mode='r+')
        self.labels = np.load(labels_path, mmap_mode='r+')

        self.num_processes = args.num_processes
        self.ks = args.ks
        self.channels = args.channels
        self.pos_grid_size = pos_grid_size
        self.stride = stride

        if args.dtype == 'float32':
            self.dtype = np.float32
        elif args.dtype == 'float16':
            self.dtype = np.float16
        else:
            raise NotImplementedError("`args.dtype` must be either `float32` or `float16`")
        
        self.buffer = ceil(data_length / self.num_processes)
        self.start_idx = self.buffer * rank

        # Precompute coordinates and positional embeddings
        C, H, W = self.images[0].shape
        new_H = (H - self.ks) // stride + 1
        new_W = (W - self.ks) // stride + 1
        row_idx = np.arange(new_H, dtype=np.int32)
        col_idx = np.arange(new_W, dtype=np.int32)
        grid_r, grid_c = np.meshgrid(row_idx, col_idx, indexing="ij")
        self.coords = np.stack([grid_r.ravel(), grid_c.ravel()], axis=1)

        # Positional embeddings
        self.pos_embedding = np.eye(pos_grid_size, dtype=self.dtype)
        self.pos_embedding_negated = 1 - self.pos_embedding


        # Base patch view setup
        self.base_shape = (self.channels, new_H, new_W, self.ks, self.ks)
        self.base_strides = (
            self.images[0].strides[0],
            self.images[0].strides[1] * stride,
            self.images[0].strides[2] * stride,
            self.images[0].strides[1],
            self.images[0].strides[2],
        )

        # Allocate buffer for full embeddings
        self.concat_buffer = np.empty((self.coords.shape[0], 4 * pos_grid_size), dtype=self.dtype)

    def __len__(self):
        return self.buffer

    def __getitem__(self, index):
        index += self.start_idx
        index = np.clip(index, 0, len(self.images)-1)
        image = self.images[index]
        label = self.labels[index]

        C, H, W = image.shape
        coords = self.coords
        pos_emb = self.pos_embedding
        neg_emb = self.pos_embedding_negated

        # 1. Patch extraction (zero-copy view)
        patches = np.lib.stride_tricks.as_strided(
            image,
            shape=self.base_shape,
            strides=self.base_strides,
            writeable=False,
        )
        patches = patches.reshape(C, -1, self.ks ** 2).transpose(1, 0, 2).reshape(-1, C * self.ks ** 2)

        # 2. Positional + negated embeddings
        np.concatenate(
            [
                pos_emb[coords[:, 0]],
                pos_emb[coords[:, 1]],
                neg_emb[coords[:, 0]],
                neg_emb[coords[:, 1]],
            ],
            axis=1,
            out=self.concat_buffer,
        )

        # 3. Combine patches and embeddings
        feat_dim = patches.shape[1] + self.concat_buffer.shape[1]
        out = np.empty((patches.shape[0], feat_dim), dtype=self.dtype)
        out[:, :patches.shape[1]] = patches
        out[:, patches.shape[1]:] = self.concat_buffer

        return torch.from_numpy(out), torch.tensor(label, dtype=torch.long)
    
def get_dataloaders(data_info, rank, args):
    train_ds = TMDataset(data_info['train_root'], data_info['train_size'], rank, args, pos_grid_size=19)
    test_ds = TMDataset(data_info['test_root'], data_info['test_size'], rank, args, pos_grid_size=19)
    pw = True if args.num_workers else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=args.pm, num_workers=args.num_workers, persistent_workers=pw)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=True, pin_memory=args.pm, num_workers=args.num_workers, persistent_workers=pw)
    return train_loader, test_loader

def get_final_test_loader(data_info, rank, args):
    final_test_ds = TMDataset(data_info['test_root'], data_info['test_size'], rank, args, pos_grid_size=19)
    test_loader = DataLoader(final_test_ds, batch_size=8, shuffle=False, pin_memory=False, num_workers=4, persistent_workers=True)
    return test_loader