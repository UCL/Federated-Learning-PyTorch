import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys
import os
import glob
from torchvision import datasets, transforms
import albumentations as A  # Data augmentation


class carvana_image_DataSet(Dataset):

    def __init__(self, data_dir: str, train: bool, transform, imgSize):
        """
        data_drir = '../data/carvana_image/train'
        """
        self.train = train
        self.transform = transform
        self.paths = self.GetPaths(data_dir)
        self.len = len(self.paths)
        self.imgSize = imgSize

    def __getitem__(self, index):
        path = self.paths[index]
        img = np.array(Image.open(path))  # Read 3-channel JPG image
        img_shape = img.shape  # Height, width and channel
        mask = np.array(Image.open(self.GetMaskPath(path)))

        sample = A.Resize(self.imgSize, self.imgSize)(
            image=img, mask=mask)

        img = sample['image']
        mask = sample['mask'].astype(np.int64)  # shifting clause

        # mask = mask[np.newaxis, :, :]  # Although mask has only one band, it also needs to add a dimension to become a three-dimensional one
        img = self.transform(img)
        return img, mask  # Returns the original data and the corresponding annotation
        # if self.train:
        #     return img, mask  # Returns the original data and the corresponding annotation
        # else:
        #     # img_shape = img.shape  # Height, width and channel
        #     # img = self.transform_test(image=img)['image']  # Data augmentation
        #     # img = get_preprocess_3band()(image=img)['image']  # Normalize the image
        #     # img_path = self.paths[index]
        #     return img

    def __len__(self):
        return self.len

    def GetPaths(self, folder):
        paths = glob.glob(os.path.join(folder, "*.jpg"))
        return paths

    def GetMaskPath(self, imgPath):
        """
        0cdf5b5d0ce1_01.jpg
        0cdf5b5d0ce1_01_mask.gif
        """
        if self.train:
            MaskPath = imgPath.replace('train', 'train_masks')
        else:
            MaskPath = imgPath.replace('test', 'test_masks')
        MaskPath = MaskPath.replace('.jpg', '_mask.gif')
        return MaskPath

if __name__ == '__main__':
    root = '../data/carvana_image/train'
    path = '../data/carvana_image/train_masks/0cdf5b5d0ce1_01_mask.gif'
    mask = np.array(Image.open(path))
    print(mask.shape)
    print(mask[640][1000])

    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    d = carvana_image_DataSet(root, True, apply_transform, 512)
    a = d[0]
    print(len(a))
    print(a[0].shape)
    print(a[1].shape)
