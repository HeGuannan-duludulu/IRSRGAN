
from utils import load_state_dict
from train_irrdb import build_model, define_optimizer, define_scheduler

rrdbnet_model, ema_rrdbnet_model = build_model()
optimizer2 = define_optimizer(rrdbnet_model)
optimizer1 = define_optimizer(rrdbnet_model)
print("Define all optimizer functions successfully.")
scheduler = define_scheduler(optimizer1)


rrdbnet_model, ema_rrdbnet_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            rrdbnet_model,
            "./test_dir/IRSRGAN_org_with_pretrained(lr=3e10-4)/g_best.pth.tar",
            ema_rrdbnet_model,
            optimizer1,
            scheduler,
            "resume")
print(start_epoch)
