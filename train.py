# encoding=utf-8

import argparse
import datetime
import os
import time

import numpy as np
import torch
# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import ImageFile
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from model_io import save_model
from modules.CONTRIQUE_model import CONTRIQUE_model, DarknetModel
from modules.configure_optimizers import configure_optimizers
from modules.dataset_loader import image_data
from modules.network import get_network, Darknet
from modules.nt_xent_multiclass import NT_Xent
from modules.sync_batchnorm import convert_model
from utils.utils import find_most_free_gpu, select_device

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')


def train(args,
          train_loader_syn,
          train_loader_ugc,
          model,
          criterion,
          optimizer,
          scaler,
          scheduler=None):
    """
    Training process
    """
    loss_epoch = 0
    model.train()

    for step, ((syn_i1, syn_i2, dist_label_syn), (ugc_i1, ugc_i2, _)) in \
            enumerate(zip(train_loader_syn, train_loader_ugc)):

        # image 1
        syn_i1 = syn_i1.cuda(non_blocking=True)
        ugc_i1 = ugc_i1.cuda(non_blocking=True)
        x_i1 = torch.cat((syn_i1, ugc_i1), dim=0)

        # image 2
        syn_i2 = syn_i2.cuda(non_blocking=True)
        ugc_i2 = ugc_i2.cuda(non_blocking=True)
        x_i2 = torch.cat((syn_i2, ugc_i2), dim=0)

        dist_label = torch.zeros((2 * args.batch_size * (args.num_patches),
                                  args.clusters + (args.batch_size * args.nodes * args.num_patches)))

        # distortion classes
        # synthetic distortion classes
        dist_label = torch.zeros((2 * args.batch_size,
                                  args.clusters + (args.batch_size * args.nodes)))
        dist_label[:args.batch_size, :args.clusters] = dist_label_syn.clone()

        # UGC data - each image is unique class
        dist_label[args.batch_size:, args.clusters + (args.nr * args.batch_size): \
                                     args.clusters + ((args.nr + 1) * args.batch_size)] = \
            torch.eye(args.batch_size)

        # all local patches inherit class of the original image
        dist_label = dist_label.repeat(1, args.num_patches).view(-1, dist_label.shape[1])
        dist_label = dist_label.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            z_i1, z_i2, z_i1_patch, z_i2_patch, h_i1, h_i2, h_i1_patch, h_i2_patch = model.forward(x_i1, x_i2)
            loss = criterion(z_i1_patch, z_i2_patch, dist_label)

        # update model weights
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print("Step [{:03d}/{:03d}]\t Loss: {:>5.3f}\t LR: {:.3f}"
                  .format(step, args.steps, loss.item(), round(lr, 5)))

        if args.nr == 0:
            args.global_step += 1

        loss_epoch += loss.item()

    return loss_epoch


def run(gpu, opt):
    """
    Run the training process
    """
    rank = opt.nr * opt.gpus + gpu

    if opt.nodes > 1:
        cur_dir = 'file://' + os.getcwd() + '/sharedfile'
        dist.init_process_group("nccl", init_method=cur_dir,
                                rank=rank, timeout=datetime.timedelta(seconds=3600),
                                world_size=opt.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    ## ---------- loader for synthetic distortions data
    train_dataset_syn = image_data(csv_path=opt.csv_file_syn,
                                   image_size=opt.image_size)
    # train_dataset_syn = PlateImageDataset(csv_path=opt.csv_file_syn,
    #                                       image_size=opt.image_size)

    if opt.nodes > 1:
        train_sampler_syn = torch.utils.data.distributed.DistributedSampler(train_dataset_syn,
                                                                            num_replicas=opt.world_size, rank=rank,
                                                                            shuffle=True)
    else:
        train_sampler_syn = None

    ## ----- Check debug or not here...
    if opt.debug:
        opt.workers = 0
    train_loader_syn = torch.utils.data.DataLoader(
        train_dataset_syn,
        batch_size=opt.batch_size,
        shuffle=(train_sampler_syn is None),
        drop_last=True,
        num_workers=opt.workers,
        sampler=train_sampler_syn,
    )

    # loader for authentically distorted data
    train_dataset_ugc = image_data(csv_path=opt.csv_file_ugc,
                                   image_size=opt.image_size)

    if opt.nodes > 1:
        train_sampler_ugc = torch.utils.data.distributed.DistributedSampler(train_dataset_ugc,
                                                                            num_replicas=opt.world_size,
                                                                            rank=rank,
                                                                            shuffle=True)
    else:
        train_sampler_ugc = None

    train_loader_ugc = torch.utils.data.DataLoader(train_dataset_ugc,
                                                   batch_size=opt.batch_size,
                                                   shuffle=(train_sampler_ugc is None),
                                                   drop_last=True,
                                                   num_workers=opt.workers,
                                                   sampler=train_sampler_ugc, )

    # initialize ResNet
    # encoder = get_network(opt.network, pretrained=False)
    # opt.n_features = encoder.fc.in_features  # get dimensions of fc layer

    encoder = Darknet(cfg_path=opt.backbone_cfg, net_size=opt.image_size)
    opt.n_features = 512  # get dimensions of fc layer

    # initialize model
    # model = CONTRIQUE_model(opt, encoder, opt.n_features)
    model = DarknetModel(opt, encoder, opt.n_features)

    # initialize model
    if opt.reload:
        ckpt_path = os.path.abspath(opt.model_path
                                    + "/checkpoint{}.tar".format(opt.epoch_num))
        if not os.path.isfile(ckpt_path):
            print("[Err]: invalid ckpt path.")
            exit(-1)
        model.load_state_dict(torch.load(ckpt_path, map_location=opt.device.type))
        print("{:s} loaded.".format(ckpt_path))
    model = model.to(opt.device)

    # sgd optimizer
    opt.steps = min(len(train_loader_syn), len(train_loader_ugc))
    opt.lr_schedule = 'warmup-anneal'
    opt.warmup = 0.1
    opt.weight_decay = 1e-4
    opt.iters = opt.steps * opt.epochs
    optimizer, scheduler = configure_optimizers(opt, model, cur_iter=-1)

    criterion = NT_Xent(opt.batch_size, opt.temperature, opt.device, opt.world_size)

    # DDP / DP
    if opt.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)

    else:
        if opt.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu]);
            print(rank);
            dist.barrier()

    model = model.to(opt.device)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    #    writer = None
    if opt.nr == 0:
        print('Training Started')

    if not os.path.isdir(opt.model_path):
        os.mkdir(opt.model_path)

    epoch_losses = []
    opt.global_step = 0
    opt.current_epoch = opt.start_epoch
    for epoch in range(opt.start_epoch, opt.epochs):
        start = time.time()

        loss_epoch = train(opt,
                           train_loader_syn,
                           train_loader_ugc,
                           model,
                           criterion,
                           optimizer,
                           scaler,
                           scheduler)

        end = time.time()
        print(np.round(end - start, 4))

        if opt.nr == 0 and epoch % 1 == 0:
            save_model(opt, model, optimizer)
            torch.save({'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, \
                       opt.model_path + 'optimizer.tar')

        if opt.nr == 0:
            print(f"Epoch [{epoch}/{opt.epochs}]\t Loss: {loss_epoch / opt.steps}")
            opt.current_epoch += 1
            epoch_losses.append(loss_epoch / opt.steps)
            np.save(opt.model_path + 'losses.npy', epoch_losses)

    ## end training
    save_model(opt, model, optimizer)


def parse_args():
    """
    Arg parser
    """
    parser = argparse.ArgumentParser(description="CONTRIQUE")
    parser.add_argument('--nodes',
                        type=int,
                        default=1,
                        help='number of nodes',
                        metavar='')
    parser.add_argument('--nr',
                        type=int,
                        default=0,
                        help='rank',
                        metavar='')
    parser.add_argument("--backbone_cfg",
                        type=str,
                        default="./yolov4_tiny_backbone.cfg",
                        help="")
    parser.add_argument('--csv_file_syn',
                        type=str,
                        default='csv_files/plates_syn.csv',
                        help='list of filenames of images with synthetic distortions')
    parser.add_argument('--csv_file_ugc',
                        type=str,
                        default='csv_files/plates_ugc.csv',
                        help='list of filenames of UGC images')
    parser.add_argument('--image_size',
                        type=tuple,
                        default=(192, 64),  # (256, 256)
                        help='image size')
    parser.add_argument('--batch_size',
                        type=int,
                        default=640,  # 32
                        help='number of images in a batch')
    parser.add_argument('--workers',
                        type=int,
                        default=8,  # 4, 8
                        help='number of workers')
    parser.add_argument("--debug",
                        type=bool,
                        default=False,
                        help="")
    parser.add_argument('--opt',
                        type=str,
                        default='sgd',
                        help='optimizer type')
    parser.add_argument('--lr',
                        type=float,
                        default=0.6,
                        help='learning rate')
    parser.add_argument('--network',
                        type=str,
                        default='resnet50',
                        help='network architecture')
    parser.add_argument('--model_path',
                        type=str,
                        default='checkpoints/',
                        help='folder to save trained models')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.1,
                        help='temperature parameter')
    parser.add_argument('--clusters',
                        type=int,
                        default=126,
                        help='number of synthetic distortion classes')
    parser.add_argument('--reload',
                        type=bool,
                        default=False,  # True
                        help='reload trained model')
    parser.add_argument('--normalize',
                        type=bool,
                        default=True,
                        help='normalize encoder output')
    parser.add_argument('--patch_dim',
                        type=tuple,
                        default=(2, 2),
                        help='number of patches for each input image')
    parser.add_argument('--projection_dim',
                        type=int,
                        default=128,
                        help='dimensions of the output feature from projector')
    parser.add_argument('--dataparallel',
                        type=bool,
                        default=False,
                        help='use dataparallel module of PyTorch')
    parser.add_argument('--start_epoch',
                        type=int,
                        default=0,
                        help='starting epoch number')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        help='total number of epochs')
    parser.add_argument("--epoch_num",
                        type=int,
                        default=0,
                        help="reloading start epoch")
    parser.add_argument('--seed',
                        type=int,
                        default=10,
                        help='random seed')
    opt = parser.parse_args()

    if opt.nodes == 1:
        ## Set up the device
        dev = str(find_most_free_gpu())
        print("[Info]: Using GPU {:s}.".format(dev))
        dev = select_device(dev)
        opt.device = dev
    else:
        opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_gpus = torch.cuda.device_count()
    opt.gpus = 1
    opt.world_size = opt.gpus * opt.nodes
    opt.num_patches = opt.patch_dim[0] * opt.patch_dim[1]

    return opt


if __name__ == "__main__":
    args = parse_args()

    if args.nodes > 1:
        print(f"Training with {args.nodes} nodes,"
              f" waiting until all nodes join before starting training")
        mp.spawn(run, args=(args,), nprocs=args.gpus, join=True)
    else:
        run(0, args)
