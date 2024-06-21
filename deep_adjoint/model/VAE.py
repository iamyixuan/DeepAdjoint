import torch
import torch.nn as nn

from ..utils.scaler import ChannelStandardScaler



class Encoder(nn.Module):
    def __init__(self, input_ch, hidden_ch, latent_dim, volume=100 * 100 * 60):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(
                input_ch, hidden_ch, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.Conv3d(
                hidden_ch, hidden_ch, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.Conv3d(
                hidden_ch, hidden_ch, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.mu = nn.Linear(int(hidden_ch * (volume / 8)), latent_dim)
        self.log_var = nn.Linear(int(hidden_ch * (volume / 8)), latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_ch, output_ch):
        super(Decoder, self).__init__()
        self.hidden_ch = hidden_ch
        self.linear = nn.Linear(latent_dim, hidden_ch * 50 * 50 * 30)
        self.decoder = nn.Sequential(
            nn.Conv3d(
                hidden_ch, hidden_ch, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.Conv3d(
                hidden_ch, hidden_ch, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.Conv3d(
                hidden_ch, hidden_ch, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_ch, output_ch, kernel_size=2, stride=2),
            nn.ReLU(),
            # nn.ConvTranspose3d(hidden_ch, output_ch, kernel_size=3, stride=1),
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(x.size(0), self.hidden_ch, 30, 50, 50)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_ch, hidden_ch, latent_dim, **kwargs):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_ch, hidden_ch, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_ch, input_ch)
        if (
            kwargs["scaler"] is not None
            and kwargs["train_data_stats"] is not None
        ):
            train_data_mean, train_data_std = kwargs["train_data_stats"]
            self.scaler = ChannelStandardScaler(
                mask=kwargs["mask"],
                mean=train_data_mean,
                std=train_data_std,
                gpu_id=kwargs["gpu_id"],
            )
            self.transform = True
        else:
            self.transform = False

    def forward(self, x):
        if self.transform:
            x = self.scaler.transform(x)
            mu, log_var = self.encoder(x)
            z = self.reparameterize(mu, log_var)
            x_hat = self.decoder(z)
            x = self.scaler.inverse_transform(x_hat)
            return x, mu, log_var
        else:
            mu, log_var = self.encoder(x)
            z = self.reparameterize(mu, log_var)
            x_hat = self.decoder(z)
            return x_hat, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
        self.reconstruct_loss = nn.MSELoss()

    def forward(self, x, x_hat, mu, log_var):
        reconstruct = self.reconstruct_loss(x, x_hat)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruct + KLD


if __name__ == "__main__":
    vae = VAE(2, 6, 16)
    x = torch.randn(1, 2, 60, 100, 100)
    x_hat, mu, log_var = vae(x)
    print(x_hat.size())
    print(mu.size())
    print(log_var.size())
