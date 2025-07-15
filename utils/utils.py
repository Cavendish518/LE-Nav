import torch
import torch.nn as nn
import random
import numpy as np
from gen_cvae import generate_data
from tqdm import tqdm

def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def re_normalize(x,device,dataset = "TEB"):
    if dataset == "TEB":
        norm_max = torch.tensor(np.array([0.4, 1.2, 0.4, 0.24, 0.24, 2.0, 2.0, 1.5, 50.0]))
        norm_min = torch.tensor(np.array([0.23, 0.59, 0.24, 0.15, 0.15, 0.99, 0.99, 0.99, 9.99]))
    elif dataset == "DWA":
        norm_max = torch.tensor(np.array([0.4, 1.2, 1.6, 2.0, 5.0, 10.0, 32.0, 0.15, 0.4]))
        norm_min = torch.tensor(np.array([0.32, 0.5, 0.6, 1.2, 2.0, 1.0, 1.0, 0.05, 0.2]))
    re_x = x*(norm_max.to(device)-norm_min.to(device)) + norm_min.to(device)
    return re_x

def poly_lr_scheduler(optimizer, init_lr, epoch, max_epoch, power=0.9):
    new_lr = init_lr * (1 - epoch / max_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def test_gen(model,input_dim,test_loader,device,top_k = 10, dataset = 'TEB'):
    total_abs_error = torch.zeros(input_dim).to(device)
    total_per_error = torch.zeros(input_dim).to(device)
    count = 0
    ave_mean_abs_error = torch.zeros(input_dim).to(device)
    ave_mean_per_error = torch.zeros(input_dim).to(device)
    for k in range(top_k):
        for ctx, x in tqdm(test_loader, desc="Evaluating"):
            ctx = ctx.to(device)
            x = x.to(device)

            model.to(device)
            with torch.no_grad():
                generated = generate_data(model, ctx, device, num_samples=1)
                generated = re_normalize(generated,device,dataset = dataset)
                x = re_normalize(x,device,dataset = dataset)
                error = generated.squeeze(0) - x.squeeze(0)
                abs_error = torch.abs(error)
                per_error = abs_error / x.squeeze(0)
                total_abs_error += abs_error
                total_per_error += per_error
                count += 1

        mean_abs_error = total_abs_error / count
        mean_per_error = total_per_error / count
        ave_mean_abs_error += mean_abs_error
        ave_mean_per_error += mean_per_error

    ave_mean_abs_error = ave_mean_abs_error / top_k
    ave_mean_per_error = ave_mean_per_error / top_k
    return ave_mean_abs_error, ave_mean_per_error