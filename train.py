import setuptools
import sys, pathlib
p = str(pathlib.Path(__file__).absolute().parents[1])
if p not in sys.path: sys.path.append(p)

import os
from pathlib import Path
from easydict import EasyDict as edict
from lib.training import Trainer
import argparse
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match nvidia-smi and CUDA device ids
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from model.training_model import TrainerModel
from model.augmenter import ImageAugmenter
from model.feature_extractor import ResnetFeatureExtractor
from model.discriminator import Discriminator
from model.seg_network import SegNetwork
import horovod.torch as hvd


class ModelParameters:

    def __init__(self, name, feature_extractor="resnet101", device="cuda:0", batch_size=None, tmodel_cache_path=None):
        """
        Model parameters;
        disc_params;
        refnet_params;
        feature_extractor;
        tmodel_cache;

        Args:
            name: the name of dir you want to save in './checkpoints' and 'tmodels_cache'
            feature_extractor: train models'name, e.g:resnet101
            device: e.g:cuda:0
            batch_size: batch size, in GPU with 11G cache, the recommended value is 8.
            tmodel_cache_path: todo: make training process short
        """
        super().__init__()

        self.name = name  # Training session name
        self.device = device
        self.batch_size = batch_size

        # Model parameters

        self.aug_params = edict(

            num_aug=15,
            min_px_count=1,

            # foreground
            fg_aug_params=edict(
                rotation=[5, -5, 10, -10, 20, -20, 30, -30, 45, -45],
                fliplr=[False, False, False, False, True],
                scale=[0.5, 0.7, 1.0, 1.5, 2.0, 2.5],
                skew=[(0.0, 0.0), (0.0, 0.0), (0.1, 0.1)],
                blur_size=[0.0, 0.0, 0.0, 2.0],
                blur_angle=[0, 45, 90, 135],
            ),
            # background
            bg_aug_params=edict(
                tcenter=[(0.5, 0.5)],
                rotation=[0, 0, 0],
                fliplr=[False],
                scale=[1.0, 1.0, 1.2],
                skew=[(0.0, 0.0)],
                blur_size=[0.0, 0.0, 1.0, 2.0, 5.0],
                blur_angle=[0, 45, 90, 135],
            ),
        )

        self.disc_params = edict(
            layer="layer4", in_channels=256 if '18' in feature_extractor else 1024, c_channels=32, out_channels=1,
            init_iters=(5, 10, 10, 10, 10), update_iters=(10,), update_filters=True,
            filter_reg=(1e-5, 1e-4), precond=(1e-5, 1e-4), precond_lr=0.1, CG_forgetting_rate=75,
            memory_size=20, train_skipping=8, learning_rate=0.1,
            pixel_weighting=None, device=self.device
        )

        self.refnet_params = edict(
            refinement_layers=["layer5", "layer4", "layer3", "layer2"],
            nchannels=64, use_batch_norm=True
        )

        self.feature_extractor = feature_extractor

        self.tmodel_cache = edict(
            enable=True,
            read_only=False,
            path=tmodel_cache_path / "{self.feature_extractor}-c{d.c_channels}".format(
                self=self, d=self.disc_params)
        )

    def get_model(self):

        augmenter = ImageAugmenter(self.aug_params)
        extractor = ResnetFeatureExtractor(name=self.feature_extractor).to(self.device)

        p = self.refnet_params
        ft_channels = {L: nch for L, nch in extractor.get_out_channels().items() if L in p.refinement_layers}
        seg_network = SegNetwork(
            in_channels=  1,
            out_channels= p.nchannels,
            ft_channels=ft_channels,
            use_bn=p.use_batch_norm)
        mdl = TrainerModel(
            augmenter=augmenter,
            feature_extractor=extractor,
            disc_params=self.disc_params,
            seg_network=seg_network,
            batch_size=self.batch_size,
            tmodel_cache=self.tmodel_cache, device=self.device)
        mdl = mdl.to(self.device)

        return mdl


if __name__ == '__main__':

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    dev = f'cuda'

    paths = dict(
        dv2017="dataset/DAVIS",
        ytvos2018="dataset/ytvos2018",
        checkpoints="checkpoints",  # Checkpoints. Session-name subdirectories will be created here.
        tensorboard="tensorboard",  # Tensorboard logs. Session-name subdirectories will be created here.
        tmcache="tmodels_cache"     # Cache of pretrained target models, requires 22 GB disk space.
    )
    # paths = dict(
    #     dv2017="~/data/datasets/DAVIS",
    #     ytvos2018="~/data/datasets/YouTubeVOS/2018",
    #     checkpoints="~/workspace/checkpoints",  # Checkpoints. Session-name subdirectories will be created here.
    #     tensorboard="~/workspace/tensorboard",  # Tensorboard logs. Session-name subdirectories will be created here.
    #     tmcache="~/camrdy_ws/tmodels_cache"     # Cache of pretrained target models, requires 22 GB disk space.
    # )
    paths = {k: Path(v).expanduser().resolve() for k, v in paths.items()}

    args_parser = argparse.ArgumentParser(description='Train FRTM')
    args_parser.add_argument('name', type=str, help='Name of the training session, for logging and saving checkpoints.')
    args_parser.add_argument('--ftext', type=str, default="resnet101", choices=["resnet101", "resnet18"], help='Feature extractor')
    args_parser.add_argument('--dset', type=str, default="all", choices=["all", "yt2018", "dv2017"],
                             help='Training datasets. all = use all data; Both DAVIS 2017 and YouTubeVOS 2018.')
    args_parser.add_argument('--dev', type=str, default="cuda:0", help='Target device to run on, default is cuda:0.')
    args_parser.add_argument('--bz', type=int, default=16, help='Batch size..')
    args = args_parser.parse_args()

    dataset = []
    if args.dset in ('all', 'dv2017'):
        dataset.append(('DAVISDataset', edict(dset_path=paths['dv2017'], epoch_repeats=8, sample_size=3)))
    if args.dset in ('all', 'yt2018'):
        dataset.append(('YouTubeVOSDataset', edict(dset_path=paths['ytvos2018'], epoch_samples=4000, min_seq_length=4, sample_size=3)))

    params = ModelParameters(args.name, feature_extractor=args.ftext, device=dev, tmodel_cache_path=paths['tmcache'], batch_size=args.bz)
    model = params.get_model()
    #only learn the parameters ofthe segmentation network, and freeze the weights of the fea-ture extractor.

    #if this params are changed, maybe not use, because 'load_stat=True' in trainer.
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=127, gamma=0.1)

    trainer = Trainer(args.name, model, optimizer=optimizer, scheduler=scheduler, dataset=dataset, checkpoints_path=paths['checkpoints'],
                      log_path=paths['tensorboard'], max_epochs=260, batch_size=params.batch_size, num_workers=8, load_latest=True, save_interval=1)
    trainer.train()

