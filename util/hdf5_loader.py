import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os

class HDF5Dataset(Dataset):
    def __init__(self, folder_path, transform=None, max_images=None):
        """
        Args:
            folder_path (string): Path to the folder containing hdf5 files.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_images (int, optional): Maximum number of images to load from the dataset.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_offsets = []
        self.total_images = 0
        self.max_images = max_images
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from HDF5 files to build an index of images."""
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith('.hdf5'):
                file_path = os.path.join(self.folder_path, file)
                with h5py.File(file_path, 'r') as hdf:
                    for video_id in hdf.keys():
                        video_group = hdf[video_id]
                        for suffix in video_group.keys():
                            suffix_group = video_group[suffix]
                            for dataset_name in suffix_group.keys():
                                dataset = suffix_group[dataset_name]
                                num_images = len(dataset)
                                # Check if adding these images would exceed the max_images limit
                                if self.max_images is not None and self.total_images + num_images > self.max_images:
                                    # Calculate how many images are needed to reach the max_images limit
                                    remaining_images = self.max_images - self.total_images
                                    if remaining_images > 0:
                                        self.image_offsets.append(
                                            (file_path, video_id, suffix, dataset_name, self.total_images, self.total_images + remaining_images)
                                        )
                                    self.total_images = self.max_images
                                    print(f"Total images: {self.total_images}")
                                    return
                                else:
                                    self.image_offsets.append(
                                        (file_path, video_id, suffix, dataset_name, self.total_images, self.total_images + num_images)
                                    )
                                    self.total_images += num_images
        print(f"Total images: {self.total_images}")

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        for file, video_id, suffix, dataset_name, start_idx, end_idx in self.image_offsets:
            if start_idx <= idx < end_idx:
                with h5py.File(file, 'r') as hdf:
                    image_idx = idx - start_idx
                    dataset = hdf[video_id][suffix][dataset_name]
                    image = dataset[image_idx]
                    break

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to [C, H, W] format

        if self.transform:
            image = self.transform(image)

        return image

    def close(self):
        pass
