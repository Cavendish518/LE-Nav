import torch

def generate_data(model, ctx, device,num_samples=1):

    model.eval()
    with torch.no_grad():
        latent_dim = model.fc_mu.out_features
        z = torch.randn(num_samples, latent_dim)
        z = z.to(device)
        ctxm = ctx[:,:4,:,:]
        mask_indices = (ctxm == -1).all(dim=2).all(dim=1)

        c = model.ctxencode(ctx,mask_indices=mask_indices)
        generated_data = model.decode(z, c)
    return generated_data

