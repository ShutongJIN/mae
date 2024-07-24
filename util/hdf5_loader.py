import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

class HDF5Dataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Path to the folder containing hdf5 files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.hdf5')]
        self.image_offsets = []
        self.total_images = 0
        for file in self.files:
            with h5py.File(file, 'r') as hdf:
                num_images = len(hdf['images'])
                self.image_offsets.append((file, self.total_images, self.total_images + num_images))
                self.total_images += num_images
        print(f"Total images: {self.total_images}")

    def __len__(self):
        print(f"Calling __len__: {self.total_images}")
        return self.total_images

    def __getitem__(self, idx):
        for file, start_idx, end_idx in self.image_offsets:
            if start_idx <= idx < end_idx:
                with h5py.File(file, 'r') as hdf:
                    image_idx = idx - start_idx
                    image = hdf['images'][image_idx]
                    break
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()  

        if self.transform:
            image = self.transform(image)

        return image

    def close(self):
        pass