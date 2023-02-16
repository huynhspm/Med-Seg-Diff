import sys
import argparse

sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.dataloader import SPMDataset
from guided_diffusion.lung_dataloader import LungDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom

viz = Visdom(port=8850)
import torchvision.transforms as transforms


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif args.data_name == 'SPM':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = SPMDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'LUNG':
        tran_list = [
            transforms.ToTensor(),
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = LungDataset(args, args.data_dir, transform_train)
        args.in_ch = 2

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args,
                       model_and_diffusion_defaults().keys()))
    if args.multi_gpu:
        model = th.nn.DataParallel(
            model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler,
                                                     diffusion,
                                                     maxt=args.diffusion_steps)

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_epoch=args.max_epoch
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',  #'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None,  #"0,1,2"
        out_dir='./results/',
        max_epoch=15)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
'''
python scripts/segmentation_train.py --data_name LUNG --data_dir ../../../data.local/vinhpt/lung-detection/LIDC_IDRI_preprocessing --out_dir ./output --image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 1
python scripts/segmentation_train.py --data_name SPM --data_dir ../../../data.local/vinhpt/MedSegDiff --out_dir ./output --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 1
'''