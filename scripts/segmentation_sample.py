import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random

sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.dataloader import SPMDataset
from guided_diffusion.lung_dataloader import LungDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset(args.data_dir, transform_test)
        args.in_ch = 5
    elif args.data_name == 'SPM':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_test = transforms.Compose(tran_list)

        ds = SPMDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4
    elif args.data_name == 'LUNG':
        tran_list = [
            transforms.ToTensor(),
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_test = transforms.Compose(tran_list)

        ds = LungDataset(args, args.data_dir, transform_test, mode='test')
        args.in_ch = 2

    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args,
                       model_and_diffusion_defaults().keys()))
    all_images = []

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    while len(all_images) * args.batch_size < args.num_samples:
        b, m, tumor, path = next(
            data)  #should return an image from the dataloader "data"

        c = th.randn_like(b[:, :1, ...])
        slice_ID = path[0].split('/')[-1].split('.')[0]
        print(slice_ID)

        # import imageio

        # try: 
        #     imageio.imsave(os.path.join(args.out_dir, str(slice_ID) + '_mask.jpg'), m[-1].moveaxis(0, 2))
        #     imageio.imsave(os.path.join(args.out_dir, str(slice_ID) + '_image.jpg'), b[-1].moveaxis(0, 2))
        # except AssertionError:
        #     cv2.imwrite(os.path.join(args.out_dir, str(slice_ID) + '_image.jpg'), b[-1].moveaxis(0, 2))

        img = th.cat((b, c), dim=1)  #add a noise channel$
        

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(
                args.num_ensemble
        ):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (diffusion.p_sample_loop_known if not args.use_ddim
                         else diffusion.ddim_sample_loop_known)
            
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(
                end))  #time measurement for the generation of 1 sample

            co = th.tensor(cal_out).repeat(1, 3, 1, 1)
            enslist.append(co)

            if args.debug:
                s = th.tensor(sample)[:, -1, :, :].unsqueeze(1).repeat(
                    1, 3, 1, 1)
                o = th.tensor(org)[:, :-1, :, :]
                c = th.tensor(cal).repeat(1, 3, 1, 1)

                tup = (o, s, c, co)

                compose = th.cat(tup, 0)
                vutils.save_image(compose,
                                  fp=os.path.join(args.out_dir, str(slice_ID) + '_output' +
                                  str(i) + ".jpg"),
                                  nrow=1,
                                  padding=10)
        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(ensres,
                          fp=os.path.join(args.out_dir, str(slice_ID) + '_output_ens' +
                          ".jpg"),
                          nrow=1,
                          padding=10)

def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,  #number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None,  #"0,1,2"
        debug=False)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()

'''
python scripts/segmentation_sample.py --data_name SPM --data_dir ../../../data.local/vinhpt/MedSegDiff --out_dir ./results --model_path ./output/savedmodel15.pt --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5
python scripts/segmentation_sample.py --data_name LUNG --data_dir ../../../data.local/vinhpt/lung-detection/LIDC_IDRI_preprocessing --out_dir ./results --model_path ./output/savedmodel05.pt --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5 --use_ddim True
'''