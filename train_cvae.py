# simple implementation
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.model import CVAE
from utils.data_ruler import FeatureTextDataset
import argparse
import yaml
import os
from datetime import datetime
from utils.utils import loss_function, set_seed, poly_lr_scheduler,test_gen
from tqdm import tqdm

def train_cvae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_all = yaml.safe_load(f)
    config = config_all["param"]
    train_dirs = config_all["train"]["img_dirs"]
    val_dirs = config_all["val"]["img_dirs"]


    seed = config.get('seed', 42)
    set_seed(seed)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    train_dataset = FeatureTextDataset(train_dirs,config['loss_aug'],config['dataset'])
    val_dataset = FeatureTextDataset(val_dirs,config['loss_aug'],config['dataset'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=config['num_workers'])

    model = CVAE(
        input_dim=config['input_dim'],
        cond_dim=config['cond_dim'],
        feed_dim=config['feed_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        mask=config['mask']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists("last_model.pth"):
        checkpoint = torch.load("last_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print("Resuming：starting from epoch", start_epoch)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('runs', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)


    model.to(device)

    best_err = 9

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        print(f"Training in epoch {epoch}")
        epoch_loss, epoch_recon, epoch_kld = 0, 0, 0
        lr = poly_lr_scheduler(optimizer, config['lr'], epoch, config['epochs'])
        writer.add_scalar('LR/train', lr, epoch)

        for i, (ctx, x) in enumerate(tqdm(train_loader)):
            ctx = ctx.to(device)
            x = x.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, ctx)

            recon_loss, kld_loss = loss_function(recon_x, x, mu, logvar)

            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_iter/Loss', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('train_iter/ReconLoss', recon_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('train_iter/KLDLoss', kld_loss.item(), epoch * len(train_loader) + i)

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld_loss.item()

        writer.add_scalar('train_epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('train_epoch/ReconLoss', epoch_recon, epoch)
        writer.add_scalar('train_epoch/KLDLoss', epoch_kld, epoch)

        # Validation
        model.eval()
        print(f"Validation in epoch {epoch}")
        val_loss, val_recon, val_kld = 0, 0, 0
        with torch.no_grad():
            for ctx, x in val_loader:
                ctx = ctx.to(device)
                x = x.to(device)

                recon_x, mu, logvar = model(x, ctx)
                recon_loss, kld_loss = loss_function(recon_x, x, mu, logvar)
                val_loss += (recon_loss + kld_loss).item()
                val_recon += recon_loss.item()
                val_kld += kld_loss.item()

        writer.add_scalar('val_epoch/Loss', val_loss, epoch)
        writer.add_scalar('val_epoch/ReconLoss', val_recon, epoch)
        writer.add_scalar('val_epoch/KLDLoss', val_kld, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, 'last_model.pth')

        if epoch > 4000:
            # start at half process to save training time
            _, ave_mean_per_error = test_gen(model, config['input_dim'], val_loader, device, top_k=10, dataset = config['dataset'])

            if ave_mean_per_error.sum() < best_err:
                best_err = ave_mean_per_error.sum()
                torch.save(model.state_dict(), 'best_model.pth')
        writer.add_scalar('val_epoch/sum_per_error', best_err, epoch)

if __name__ == "__main__":
    train_cvae()
