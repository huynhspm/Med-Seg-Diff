import os.path as osp
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import imageio
import cv2


class LungDataset(Dataset):

    def __init__(self,
                 args,
                 data_dir,
                 transform=None,
                 mode='train',
                 plane=False):

        config_data_dir = osp.join(data_dir, 'config_data', mode + '.csv')
        self.mode = mode
        self.data_dir = data_dir
        self.data_list = pd.read_csv(config_data_dir, names=['image', 'mask', 'tumor'])
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """Get the images"""
        image = self.data_list['image'][index]
        mask = self.data_list['mask'][index]
        tumor = self.data_list['tumor'][index]

        if self.mode == 'test':
            img = image.split('/')
            image = osp.join('data/Segment/Image', img[-2], img[-1])
            mask = osp.join('data/Segment/Mask', img[-2], img[-1])

        image_path = osp.join(self.data_dir, image)
        mask_path = osp.join(self.data_dir, mask)

        image = np.load(image_path)
        mask = np.load(mask_path)

        try: 
            imageio.imsave('image.jpg', image)
        except AssertionError:
            cv2.imwrite('image.jpg', image)
    
        image = imageio.v2.imread('./image.jpg')

        mask = [[int(x) for x in row] for row in mask]
        mask = np.array(mask)
        mask = torch.from_numpy(mask).unsqueeze(0)

        imageio.imsave('mask.jpg', mask)
        mask = imageio.v2.imread('./mask.jpg')
        
        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        # file_name = osp.join('test/gt', mask_path.split('/')[-1][:-3] + 'jpg')
        # print(file_name)    
        # imageio.imsave(file_name, mask.squeeze())
        
        # return 0
        return (image, mask, tumor, image_path)


if __name__ == '__main__':
    from torchvision import transforms as T

    dataset = LungDataset(None, data_dir='../../../data.local/vinhpt/lung-detection/LIDC_IDRI_preprocessing', 
                          mode='test',
                          transform=T.Resize((128, 128)))
    print(len(dataset))

    # for i in range(len(dataset)):
    #     a = dataset[i]

    # print(image.shape)
    # print(mask.shape)
    # print(tumor)

    # mask = mask.moveaxis(0, 2)
    # imageio.imsave('mask.jpg', mask)

    # image = image.moveaxis(0, 2)
    # imageio.imsave('image.jpg', image)