import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import irrdbnet_config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_evaluate import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    """构建模型，这个函数要改一下,适配我自己的网络模型"""
    rrdbnet_model, ema_rrdbnet_model = build_model()
    print(f"Build `{irrdbnet_config.g_arch_name}` model successfully.")

    """L1 loss, 不明白为啥是这个"""
    criterion = define_loss()
    print("Define all loss functions successfully.")

    """为生成器量身定制的Adam优化函数和scheduler函数微调"""
    optimizer = define_optimizer(rrdbnet_model)
    print("Define all optimizer functions successfully.")
    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    """是否加载预训练模型"""
    print("Check whether to load pretrained model weights...")
    if irrdbnet_config.pretrained_g_model_weights_path:
        rrdbnet_model = load_state_dict(
            rrdbnet_model,
            irrdbnet_config.pretrained_g_model_weights_path
        )
        print(f"Loaded `{irrdbnet_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    """从训练好的预训练模型继续训练"""
    print("Check whether the pretrained model is restored...")
    if irrdbnet_config.resume_g_model_weights_path:
        rrdbnet_model, ema_rrdbnet_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            rrdbnet_model,
            irrdbnet_config.pretrained_g_model_weights_path,
            ema_rrdbnet_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results, 保存实验用例和结果 smaples/irrdbnet_config.exp_name; results/irrdbnet_config.exp_name
    samples_dir = os.path.join("samples", irrdbnet_config.exp_name)
    results_dir = os.path.join("results", irrdbnet_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file, 保存log文件 samples/logs
    writer = SummaryWriter(os.path.join("samples", "logs", irrdbnet_config.exp_name))

    # Initialize the gradient scaler， 初始化导数
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    """这里 第二个参数没太看懂，不知道什么意思"""
    psnr_model = PSNR(irrdbnet_config.upscale_factor, irrdbnet_config.only_test_y_channel)
    ssim_model = SSIM(irrdbnet_config.upscale_factor, irrdbnet_config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=irrdbnet_config.device)
    ssim_model = ssim_model.to(device=irrdbnet_config.device)

    """进行这么多轮训练"""
    for epoch in range(start_epoch, irrdbnet_config.epochs):
        train(rrdbnet_model,
              ema_rrdbnet_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer)
        psnr, ssim = validate(rrdbnet_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == irrdbnet_config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": rrdbnet_model.state_dict(),
                         "ema_state_dict": ema_rrdbnet_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    """加载数据集"""
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(irrdbnet_config.train_gt_images_dir,
                                            irrdbnet_config.gt_image_size,
                                            irrdbnet_config.upscale_factor,
                                            "Train")

    test_datasets = TestImageDataset(irrdbnet_config.test_gt_images_dir, irrdbnet_config.test_lr_images_dir)

    """将加载的数据集装载到DataLoader"""
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=irrdbnet_config.batch_size,
                                  shuffle=True,
                                  num_workers=irrdbnet_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, irrdbnet_config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, irrdbnet_config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    """通过测试，可以成功导入自己构建的网络"""
    rrdbnet_model = model.Generator()
    # load model in cpu or gpu
    rrdbnet_model = rrdbnet_model.to(device=irrdbnet_config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - irrdbnet_config.model_ema_decay) * \
                                                                              averaged_model_parameter + irrdbnet_config.model_ema_decay * \
                                                                              model_parameter
    ema_rrdbnet_model = AveragedModel(rrdbnet_model, avg_fn=ema_avg)

    return rrdbnet_model, ema_rrdbnet_model


def define_loss() -> nn.L1Loss:
    """定义了loss函数，但我不明白为啥生成器是l1 loss"""
    criterion = nn.L1Loss()
    criterion = criterion.to(device=irrdbnet_config.device)

    return criterion


def define_optimizer(rrdbnet_model) -> optim.Adam:
    """Adam优化器：
        参数：model_lr = 2e-4, 说3e-4更好？
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0
        """
    optimizer = optim.Adam(rrdbnet_model.parameters(),
                           irrdbnet_config.model_lr,
                           irrdbnet_config.model_betas,
                           irrdbnet_config.model_eps,
                           irrdbnet_config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    """设置scheduler，动态调整学习率"""
    scheduler = lr_scheduler.StepLR(optimizer,
                                    irrdbnet_config.lr_scheduler_step_size,
                                    irrdbnet_config.lr_scheduler_gamma)

    return scheduler


def train(
        rrdbnet_model: nn.Module,
        ema_rrdbnet_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    """进入训练模式"""
    rrdbnet_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt: torch.Tensor = batch_data["gt"].to(device=irrdbnet_config.device, non_blocking=True)
        lr: torch.Tensor = batch_data["lr"].to(device=irrdbnet_config.device, non_blocking=True)

        # Initialize generator gradients
        rrdbnet_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = rrdbnet_model(lr)
            """calculating the loss between ground-truth and sr """
            loss = torch.mul(
                irrdbnet_config.loss_weights,
                criterion(sr, gt)
            )

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        """我不知道这是干啥的"""
        ema_rrdbnet_model.update_parameters(rrdbnet_model)

        # Statistical loss value for terminal data output
        """输出的损失值loss"""
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        """到一定周期，就打印记录数值"""
        if batch_index % irrdbnet_config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        """下一个batch_data"""
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        rrdbnet_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    rrdbnet_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=irrdbnet_config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=irrdbnet_config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = rrdbnet_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % irrdbnet_config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
