import os.path as osp
import glob
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

class SPMDataset(Dataset):

    def __init__(self,
                 args,
                 data_path,
                 transform=None,
                 mode='Training',
                 plane=False):

        input_data_dir = osp.join(data_path,
                                  'ISIC2018_Task1-2_' + mode + '_Input')
        gt_data_dir = osp.join(data_path,
                               'ISIC2018_Task1_' + mode + '_GroundTruth')

        img_list = glob.glob(osp.join(input_data_dir, '*.jpg'))
        # gt_list = glob.glob(osp.join(gt_data_dir, '*.png'))
        # assert (len(img_list) == len(gt_list))

        self.data_path = data_path
        self.mode = mode
        self.name_list = [name.split('/')[-1][0:-4] for name in img_list]
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]

        img_path = osp.join(self.data_path,
                            'ISIC2018_Task1-2_' + self.mode + '_Input',
                            name + '.jpg')

        gt_path = osp.join(self.data_path,
                           'ISIC2018_Task1_' + self.mode + '_GroundTruth',
                           name + '_segmentation.png')

        img = Image.open(img_path)
        

        if self.mode == 'Training':
            mask = Image.open(gt_path)
        else:
            mask = img
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        if self.mode == 'Training':
            return (img, mask)
        else:
            return (img, mask, img_path)


if __name__ == '__main__':

    from torchvision import transforms as T
    dataset = SPMDataset(None, 
                        data_path='../../../data.local/vinhpt/MedSegDiff', 
                        mode='Training', 
                        transform=T.ToTensor())
    print(len(dataset))
    img, mask = dataset[1]
    print(img.shape)
    print(mask.shape)

    # plt.imshow(img)
    # plt.show()

    # plt.imshow(mask)
    # plt.show()