import torch
import torch.nn as nn
from utils import _init_weight, device, ModelArgs

args = ModelArgs()


class Discriminator(nn.Module):
    def __init__(self,device=device):
        super(Discriminator, self).__init__()
        self.n_actions = args.n_continuous_action + args.n_discrete_action
        self.n_state = args.n_discrete_state + args.n_continuous_state
        n_discriminator_hidden = args.n_discriminator_hidden
        self.discriminator_net = nn.Sequential(
            nn.Linear(self.n_state + self.n_actions, n_discriminator_hidden),
            nn.Tanh(),
            nn.Linear(n_discriminator_hidden, n_discriminator_hidden),
            nn.Tanh(),
            nn.Linear(n_discriminator_hidden, 1),
            nn.Sigmoid()
        ).to(device)
        self.discriminator_net.apply(_init_weight)

    def forward(self, state, action):
        state_action = torch.cat((state.to(device), action.to(device)), dim=-1)
        x = self.discriminator_net(state_action.to(device))
        return x.to(device)


__all__ = ['Discriminator']
