import torch
import torch.nn as nn

# Discriminator
class WGANCritic(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(WGANCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, 1)  # Output a single scalar value
        )

    def forward(self, x):
        return self.model(x)


class WGANLoss:
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp
        self.gp = None

    def gradient_penalty(self, critic, source_samples, target_samples):
        """
        Required to satisfy Lipschitz-1 condition
        """
        batch_size = source_samples.size(0)
        alpha = torch.rand(batch_size, 1).to(source_samples.device)  # Random weight for interpolation
        alpha = alpha.expand_as(source_samples)
        interpolated = alpha * source_samples + (1 - alpha) * target_samples
        interpolated.requires_grad_(True)

        critic_interpolated = critic(interpolated)

        grad_outputs = torch.ones_like(critic_interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        self.gp = gp * self.lambda_gp
        return gp * self.lambda_gp

    def discrinimative(self, critic_target, critic_source, gp):
        loss_critic = critic_target.mean() - critic_source.mean() + gp
        return loss_critic

    def generative(self, critic_target):
        loss_generative = -critic_target.mean()
        return loss_generative

    def __call__(self, critic, source_samples, target_samples, mode='d'):
        if mode == 'd':
            gp = self.gradient_penalty(critic, source_samples, target_samples)
            critic_source = critic(source_samples)
            critic_target = critic(target_samples)
            loss = self.discrinimative(critic_target, critic_source, gp)
        elif mode == 'g':
            critic_target = critic(target_samples)
            loss = self.generative(critic_target)
        return loss
        

