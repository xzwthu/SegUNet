import random
import torch 
from PIL import Image
from torch.utils import data 
from torchvision import transforms
import os.path as osp
import numpy as np
import SimpleITK as sitk
import imgaug as ia
import imgaug.augmenters as iaa
from scipy.ndimage import zoom

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.2, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5),
        sometimes(iaa.Affine(
            scale={"x": (1, 1.05), "y": (1, 1.05)}, # scale images to 80-120% of their size, individually per axis
            rotate=(-5, 5), # rotate by -45 to +45 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 3 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        # iaa.SomeOf((0, 3),
        #     [
        #         iaa.OneOf([
        #             iaa.GaussianBlur((0.0, 0.8)), # blur images with a sigma between 0 and 0.8
        #             iaa.AverageBlur(k=(0, 3)), # blur image using local means with kernel sizes between 0 and 3
        #         ]),
        #         # iaa.Sharpen(alpha=(0.01,0.01), lightness=(0.95, 1.00)), # sharpen images
        #         # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5), # add gaussian noise to images
        #         sometimes(iaa.ElasticTransformation(alpha=(0.0, 0.4), sigma=0.0005)), # move pixels locally around (with random strengths)
        #         sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03)))
        #     ],
        #     random_order=True
        # )
        sometimes(
            iaa.OneOf([
                iaa.GaussianBlur((0.0, 0.8)), # blur images with a sigma between 0 and 0.8
                iaa.AverageBlur(k=(0, 3)), # blur image using local means with kernel sizes between 0 and 3
            ])
        ),
        sometimes(iaa.ElasticTransformation(alpha=(0.0, 0.4), sigma=0.0005)), # move pixels locally around (with random strengths)
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.03)))
    ],
    random_order=True
)
class myDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        ignore_label,
        crop_size,
        augment = True,
    ):
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.augment = augment
        self.crop_size = crop_size
        self.ids = []
        self._set_files()
        
    def _set_augment(self,augment=False):
        self.augment = augment
    def _set_image_dir(self,image_dir = 'image'):
        self.image_dir = osp.join(self.root,image_dir)
    def _set_files(self):
        self.image_dir = osp.join(self.root,'image')
        self.label_dir = osp.join(self.root,'label')
        self.mask_dir = osp.join(self.root,'label')
        file_list = osp.join(
            self.root,'id_txt',self.split +'.txt'
        )
        with open(file_list) as f:
            contents = f.readlines()
        self.ids = [n[:-1] for n in contents]
    def _augmentation(self,image,label):
        #TODO multi-scale resize
        d,h,w = image.shape
        assert len(image.shape) == len(self.crop_size)
        start_d = random.randint(0,d-self.crop_size[0])
        end_d = int(start_d+self.crop_size[0])
        start_h = random.randint(0,h-self.crop_size[1])
        end_h = int(start_h+self.crop_size[1])
        start_w = random.randint(0,w-self.crop_size[2])
        end_w = int(start_w+self.crop_size[2])

        # import pdb; pdb.set_trace()
        image = image[start_d:end_d,start_h:end_h,start_w:end_w]
        noise_ = np.random.random(image.shape)*0.01-0.005
        image += noise_
        if label is not None:
            label = label[start_d:end_d,start_h:end_h,start_w:end_w]
        return image, label
    def __getitem__(self, index):
        image_id = self.ids[index]
        image_path = osp.join(self.image_dir, image_id + '.nii.gz')
        img = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(img)
        label_path = osp.join(self.label_dir,image_id+'.nii.gz')
        img = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(img)
        image = (image.astype(np.float32)-image.min())/(image.max()-image.min())
        d,h,w = image.shape
        #TODO image size parameter
        image = zoom(image,(1,256/h,256/w))
        label = zoom(label,(1,256/h,256/w),order = 0)
        if self.augment:
            image, label = self._augmentation(image,label)
        image = image[np.newaxis,:,:]
        # image = image*mask
        # if ('_wL' in self.split) or ('_H' in self.split):
        #     image = image.transpose(0,2,3,1)
        #     mask = mask.transpose(0,2,3,1).astype('uint8')
        #     image,mask = seq(images = image,segmentation_maps= mask)
        #     image = image.transpose(0,3,1,2)
        #     mask = mask.transpose(0,3,1,2).astype('float32')
        # if ('_L' in self.split):
        #     image = image.transpose(0,2,3,1)
        #     label = label[None].transpose(0,2,3,1)
        #     image, label = seq(images = image,segmentation_maps = label)
        #     image = image.transpose(0,3,1,2)
        #     label = label.transpose(0,3,1,2)
        #     label = label[0]
        image[image>1]=1
        image[image<0]=0
        image = image*2-1
        return image_id,image,label.astype('float32')
    def __len__(self):
        return len(self.ids)
    
        
    
