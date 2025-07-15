# simple implementation
import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, feed_dim, latent_dim, hidden_dim,mask =False):
        super(CVAE, self).__init__()
        self.positional_encoding = PositionalEncoding(cond_dim, max_len=500)

        self.mask = mask
        self.C_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cond_dim, nhead=1, dim_feedforward=feed_dim,batch_first=True),
            num_layers=2
        )

        self.C_linear = nn.Sequential(
            nn.Linear(cond_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def generate_mask(self, batch_size, seq_len, device):
        mask = torch.zeros([batch_size, seq_len], dtype=torch.bool)
        num_masked = torch.randint(0, seq_len, (batch_size,))
        for i in range(batch_size):
            mask_indices = torch.randperm(seq_len)[:num_masked[i]]
            mask[i, mask_indices] = True
        return mask.to(device)

    def ctxencode(self, ctx, mask_indices=None):

        batch_size, dim1, dim2, seq_len = ctx.size()
        ctx = ctx.view(batch_size, dim1 * dim2, seq_len).permute(0,2,1)

        ctx = self.positional_encoding(ctx)

        if self.mask:
            if mask_indices is not None:
                ctx_transformed = self.C_transformer(
                    ctx,
                    src_key_padding_mask=mask_indices
                )

            else:
                mask = self.generate_mask(batch_size=batch_size, seq_len=seq_len,device = ctx.device)
                ctx_transformed = self.C_transformer(
                    ctx,
                    src_key_padding_mask=mask
                )

        else:
            ctx_transformed = self.C_transformer(
                ctx
            )

        ctx_out = self.C_linear(ctx_transformed.reshape(batch_size, seq_len * dim1 * dim2))

        return ctx_out

    def encode(self, x, c):
        x_cond = torch.cat([x, c], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))


    def forward(self, x, ctx):
        ctxm = ctx[:,:4,:,:]
        input_mask = (ctxm == -1).all(dim=2).all(dim=1)

        c = self.ctxencode(ctx,mask_indices=input_mask)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

