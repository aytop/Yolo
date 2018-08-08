from torchvision import transforms
import torch.utils.data as ds
import numpy as np
import PIL.Image as Image
import torch

import pathlib


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = np.array(Image.fromarray(image).resize((new_h, new_w)))
        return {'image': img, 'label': label}


class DetectionDataSet(ds.Dataset):
    def __init__(self, paths='numpy/paths.txt', label_dir='numpy/', root_dir='train/',
                 transform=transforms.Compose([Rescale((144, 72)), ToTensor()]),
                 abs_path='/home/aytop/PycharmProjects/Praksa/'):
        """
                Args:
                    paths (string, optional): Textual file containing paths to numpy labels.
                    label_dir (string, optional): Path to the directory with numpy labels.
                    root_dir (string, optional): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
        """
        self.paths = abs_path + paths
        self.label_dir = abs_path + label_dir
        self.root_dir = abs_path + root_dir
        self.transform = transform

    def __len__(self):
        return sum(1 for _ in open(self.paths))

    def __getitem__(self, item):
        index = -1
        with open(self.paths) as paths:
            for i, p in enumerate(paths):
                if i == item:
                    index = p.split('/')[1].strip()
        image = np.array(Image.open(self.root_dir+index+'.png'))
        label = np.load(self.label_dir+index+'.npy')
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
