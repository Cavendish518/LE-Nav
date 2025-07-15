import torch
from torch.utils.data import DataLoader
import yaml
import os
from model import CVAE
from data_ruler import FeatureTextDataset,FeatureTextDataset_los
from gen_cvae import generate_data
import numpy as np
from tqdm import tqdm

def re_normalize(x,dataset = "TEB"):
    if dataset == "TEB":
        norm_max = torch.tensor(np.array([0.4, 1.2, 0.4, 0.24, 0.24, 2.0, 2.0, 1.5, 50.0]))
        # norm_min = torch.tensor(np.array([0.24, 0.6, 0.25, 0.16, 0.16, 1.0, 1.0, 1.0, 10.0]))
        norm_min = torch.tensor(np.array([0.23, 0.59, 0.24, 0.15, 0.15, 0.99, 0.99, 0.99, 9.99]))
    elif dataset == "DWA":
        norm_max = torch.tensor(np.array([0.4, 1.2, 1.6, 2.0, 5.0, 10.0, 32.0, 0.15, 0.4]))
        norm_min = torch.tensor(np.array([0.32, 0.5, 0.6, 1.2, 2.0, 1.0, 1.0, 0.05, 0.2]))
    re_x = x*(norm_max.to("cuda:0")-norm_min.to("cuda:0")) + norm_min.to("cuda:0")
    return re_x

def evaluate_generation(top_k=1):
    # 读取配置
    with open('config.yaml', 'r') as f:
        config_all = yaml.safe_load(f)
    config = config_all["param"]
    test_dirs = config_all["test"]["img_dirs"]  # 默认使用验证集作为测试集

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    # 加载数据
    # test_dataset = FeatureTextDataset(test_dirs, config['loss_aug'])
    test_dataset = FeatureTextDataset_los(test_dirs, config['loss_aug'],config['dataset'])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

    # 模型加载
    model = CVAE(
        input_dim=config['input_dim'],
        cond_dim=config['cond_dim'],
        feed_dim=config['feed_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        mask=config['mask']
    ).to(device)

    model_path = "best_model.pth"
    assert os.path.exists(model_path), f"{model_path} 不存在，请先训练模型。"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 初始化误差累积容器
    input_dim = config['input_dim']
    total_abs_error = torch.zeros(input_dim).to(device)
    total_per_error = torch.zeros(input_dim).to(device)
    count = 0
    top_mean_abs_error = torch.zeros(input_dim).to(device)
    top_mean_per_error = torch.zeros(input_dim).to(device)
    ave_mean_abs_error = torch.zeros(input_dim).to(device)
    ave_mean_per_error = torch.zeros(input_dim).to(device)
    for k in range(top_k) :
        for ctx, x in tqdm(test_loader, desc="Evaluating"):
            ctx = ctx.to(device)
            x = x.to(device)

            model.to(device)
            with torch.no_grad():
                generated = generate_data(model, ctx, device, num_samples=1)
                generated = re_normalize(generated, dataset=config['dataset'])
                x = re_normalize(x, dataset=config['dataset'])
                error = generated.squeeze(0) - x.squeeze(0)
                # print("reset")
                # print(x.squeeze(0))
                # print(generated.squeeze(0))
                # print(error)
                abs_error = torch.abs(error)  # (input_dim,)
                per_error = abs_error / x.squeeze(0)
                total_abs_error += abs_error
                total_per_error += per_error
                count += 1

        mean_abs_error = total_abs_error / count
        mean_per_error = total_per_error / count
        ave_mean_abs_error += mean_abs_error
        ave_mean_per_error += mean_per_error
        if k == 0:
            top_mean_abs_error = mean_abs_error
            top_mean_per_error = mean_per_error
        else:
            if top_mean_per_error.sum() > mean_per_error.sum():
                top_mean_abs_error = mean_abs_error
                top_mean_per_error = mean_per_error
    ave_mean_abs_error = ave_mean_abs_error / top_k
    ave_mean_per_error = ave_mean_per_error / top_k
    # 打印每个维度的平均误差
    print("Per-dimension ave Mean Absolute Error:")
    for i, err in enumerate(ave_mean_abs_error.cpu().numpy()):
        print(f"  Dim {i}: {err:.6f}")
    print("Per-dimension ave Mean percentage Error:")
    for i, err in enumerate(ave_mean_per_error.cpu().numpy()):
        print(f"  Dim {i}: {err:.6f}")
    print("Per-dimension top Mean Absolute Error:")
    for i, err in enumerate(top_mean_abs_error.cpu().numpy()):
        print(f"  Dim {i}: {err:.6f}")
    print("Per-dimension top Mean percentage Error:")
    for i, err in enumerate(top_mean_per_error.cpu().numpy()):
        print(f"  Dim {i}: {err:.6f}")

    print(f"top err:{top_mean_per_error.mean()}")
    print(f"mean err:{ave_mean_per_error.mean()}")

if __name__ == "__main__":
    top_k = 10 #10
    evaluate_generation(top_k=top_k)
