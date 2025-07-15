from torch.utils.data import Dataset
import numpy as np
import os
import torch

class FeatureTextDataset(Dataset):
    def __init__(self, dirs, loss_aug, dataset = "TEB"):
        self.normalize = True
        self.dataset = dataset
        if self.dataset == "TEB":
            self.norm_max = np.array([0.4, 1.2, 0.4, 0.24, 0.24,  2.0, 2.0, 1.5, 50.0])
            self.norm_min = np.array([0.23, 0.59, 0.24, 0.15, 0.15, 0.99, 0.99, 0.99, 9.99]) # for boundary
        elif self.dataset == "DWA":
            self.norm_max = np.array([0.4, 1.2, 1.6, 2.0, 5.0, 10.0, 32.0, 0.15, 0.4])
            self.norm_min = np.array([0.32, 0.5, 0.6, 1.2, 2.0, 1.0, 1.0, 0.05, 0.2])

        self.samples = []
        self.loss_aug = loss_aug

        if isinstance(dirs, str):
            dirs = [dirs]

        for root_dir in dirs:
            feature_path = os.path.join(root_dir, 'features.npz')
            text_path = os.path.join(root_dir, 'output.txt')

            if not os.path.exists(feature_path) or not os.path.exists(text_path):
                print(f"{root_dir} lacks features.npz or output.txt")
                continue

            features = np.load(feature_path)
            with open(text_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            if len(features.files) != len(lines):
                print(f"{root_dir} error")
                continue

            for idx in range(len(lines)):
                feat_key = f'feat_{idx:06d}'
                org_feature = features[feat_key]
                context_features = []
                for offset in range(2, -1, -1):
                    prev_idx = idx - offset
                    if prev_idx < 0:
                        context_features.append(-1*np.ones_like(org_feature))
                    else:
                        prev_feat_key = f'feat_{prev_idx:06d}'
                        context_features.append(features[prev_feat_key])

                feature = np.stack(context_features, axis=-1)
                text = np.array(list(map(float, lines[idx].strip('[]').split(', '))))[1:]
                if self.normalize:
                    text = (text - self.norm_min)/(self.norm_max - self.norm_min)
                self.samples.append((feature, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature, text = self.samples[idx]
        if self.loss_aug:
            feature = self.loss_sim(feature)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(text, dtype=torch.float32)

    def loss_sim(self, feature):
        if np.any(feature == -1):
            return feature
        else:
            num_indices = np.random.randint(0, 2)
            indices = np.random.choice(3, num_indices, replace=False)
            for idx in indices:
                feature[:4, :, idx] = -1
        return feature
