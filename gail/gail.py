import math

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter

from model.critic import Value
from model.discriminator import Discriminator
from model.policy import Policy
from utils import ModelArgs, device

import time

args = ModelArgs()
# discrete sections of state and action
discrete_action_sections = [5, 4, 11]
discrete_state_sections = [10, 3, 16, 4]


def GAE(reward, value, mask, gamma, lam):
    adv = torch.FloatTensor(reward.shape[0], 1).to(device)
    delta = torch.FloatTensor(reward.shape[0], 1).to(device)

    pre_value, pre_adv = 0, 0
    for i in reversed(range(reward.shape[0])):
        delta[i] = reward[i] + gamma * pre_value * mask[i] - value[i]
        adv[i] = delta[i] + gamma * lam * pre_adv * mask[i]
        pre_adv = adv[i, 0].to(device)
        pre_value = value[i, 0].to(device)
    returns = value.to(device) + adv.to(device)
    adv = (adv - adv.mean()) / adv.std()
    return adv.to(device), returns.to(device)


def ppo_step(policy, value, optimizer_policy, optimizer_value, discrete_state, continuous_state, discrete_action,
             continuous_action, next_discrete_state, next_continuous_state, reward,
             fixed_log_probs, done, ppo_clip_epsilon):
    # update critic
    states = torch.cat((discrete_state, continuous_state), dim=-1).to(device)
    actions = torch.cat((discrete_action, continuous_action), dim=-1).to(device)
    with torch.no_grad():
        values_pred = value(states.to(device))
    advantages, returns = GAE(reward.detach(), values_pred.detach(), done, gamma=args.gamma, lam=args.lam)
    value_loss = (values_pred - returns).pow(2).mean()
    for param in value.parameters():
        value_loss += param.to(device).pow(2).sum() * 1e-3
    optimizer_value.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), 40)
    optimizer_value.step()

    # update actor
    policy_log_probs = policy.get_policy_net_log_prob(torch.cat((discrete_state, continuous_state), dim=-1), discrete_action,
                                               continuous_action)
    transition_log_prob_new = policy.get_transition_net_log_prob(
        torch.cat((discrete_state, continuous_state, discrete_action, continuous_action), dim=-1), next_discrete_state,
        next_continuous_state)
    log_probs = policy_log_probs + transition_log_prob_new
    ratio = torch.exp(log_probs.to(device) - fixed_log_probs)
    # print('log_probs', log_probs)
    # print('fixed_log_probs', fixed_log_probs)
    # print(ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
    policy_loss = -torch.min(surr1.to(device), surr2.to(device)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 40)
    optimizer_policy.step()
    return value_loss, policy_loss


class ExpertDataSet(torch.utils.data.Dataset):
    def __init__(self, activity_file_name, cost_file_name):
        self.a = torch.Tensor(pd.read_json(activity_file_name).data)
        self.c = torch.Tensor(pd.read_json(cost_file_name).data)
        self.a = torch.Tensor(self.normalization(self.a))
        self.c = torch.Tensor(self.normalization(self.c))
        self.length = len(self.a)
        self.a_shape = self.a[0].shape
        self.c_shape = self.c[0].shape
        self.init_state = torch.randn(self.c_shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.c[idx], self.a[idx]

    def normalization(self, x):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        return x


def main():
    # define actor/critic/discriminator net and optimizer
    policy = Policy(discrete_action_sections, discrete_state_sections)
    value = Value().to(device)
    discriminator = Discriminator()
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=args.value_lr)
    optimizer_value = torch.optim.Adam(value.parameters(), lr=args.policy_lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.discrim_lr)
    discriminator_criterion = nn.BCELoss()

    # # load expert data
    # dataset = ExpertDataSet(args.expert_activities_data_path, args.expert_cost_data_path)
    # data_loader = Data.DataLoader(
    #     dataset=dataset,
    #     batch_size=args.expert_batch_size,
    #     shuffle=False,
    #     num_workers=1
    # )

    # load models
    # discriminator.load_state_dict(torch.load(''))
    # policy.transition_net.load_state_dict(torch.load(''))
    # policy.policy_net.load_state_dict(torch.load(''))
    # value.load_state_dict(torch.load(''))

    # update discriminator
    for _ in range(args.training_epochs):
        # collect data from environment for ppo update
        memory = policy.collect_samples(args.ppo_buffer_size)
        batch = memory.sample()
        discrete_state = torch.stack(batch.discrete_state).to(device).squeeze_().detach()
        continuous_state = torch.stack(batch.continuous_state).to(device).squeeze_().detach()
        discrete_action = torch.stack(batch.discrete_action).to(device).squeeze_().detach()
        continuous_action = torch.stack(batch.continuous_action).to(device).squeeze_().detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).to(device).squeeze_().detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).to(device).squeeze_().detach()
        old_log_prob = torch.stack(batch.old_log_prob).to(device).squeeze_().detach()
        mask = torch.Tensor(batch.mask).to(device).detach()

        # for test
        expert_state_batch = torch.load('./exp_s_batch')
        expert_action_batch = torch.load('./exp_a_batch')
        gen_r = discriminator(
            torch.cat((discrete_state, continuous_state), dim=-1),
            torch.cat((discrete_action, continuous_action), dim=-1))
        expert_r = discriminator(expert_state_batch.to(device), expert_action_batch.to(device))
        optimizer_discriminator.zero_grad()
        d_loss = discriminator_criterion(gen_r.to(device),
                                         torch.zeros(gen_r.shape, device=device)) + \
                 discriminator_criterion(expert_r.to(device),
                                         torch.ones(expert_r.shape, device=device).to(device))
        d_loss.backward()
        optimizer_discriminator.step()

        '''
        for _ in range(1):
            for expert_state_batch, expert_action_batch in data_loader:
                gen_r = discriminator(
                    torch.cat((discrete_state, continuous_state), dim=-1),
                    torch.cat((discrete_action, continuous_action), dim=-1))
                expert_r = discriminator(expert_state_batch.to(device), expert_action_batch.to(device))
                optimizer_discriminator.zero_grad()
                d_loss = discriminator_criterion(gen_r.to(device),
                                                 torch.zeros(gen_r.shape, device=device)) + \
                         discriminator_criterion(expert_r.to(device),
                                                 torch.ones(expert_r.shape, device=device).to(device))
                d_loss.backward()
                optimizer_discriminator.step()
        '''


        # update PPO
        optimize_iter_num = int(math.ceil(discrete_state.shape[0] / args.ppo_mini_batch_size))
        for _ in range(args.ppo_optim_epoch):
            for i in range(optimize_iter_num):
                index = slice(i * args.ppo_mini_batch_size,
                              min((i + 1) * args.ppo_mini_batch_size, discrete_state.shape[0]))
                discrete_state_batch, continuous_state_batch, discrete_action_batch, continuous_action_batch, \
                gen_r_batch, old_log_prob_batch, mask_batch, next_discrete_state_batch, next_continuous_state_batch = \
                    discrete_state[index], continuous_state[index], discrete_action[index], continuous_action[index], \
                    gen_r[index], old_log_prob[index], mask[index], next_discrete_state[index], next_continuous_state[index]

                v_loss, p_loss = ppo_step(policy, value, optimizer_policy, optimizer_value,
                                          discrete_state_batch.to(device),
                                          continuous_state_batch.to(device),
                                          discrete_action_batch.to(device), continuous_action_batch.to(device), next_discrete_state_batch.to(device),
                                          next_continuous_state_batch.to(device),
                                          gen_r_batch.to(device), old_log_prob_batch.to(device),
                                          mask_batch.to(device), args.ppo_clip_epsilon)
                # print('d_loss', d_loss)
                # print('p_loss', p_loss)
                # print('v_loss', v_loss)
            # save models
        torch.save(discriminator.state_dict(), './{}:Discriminator_model.pkl'.format(args.ppo_optim_epoch))
        torch.save(policy.transition_net.state_dict(), './{}:Transition_model.pkl'.format(args.ppo_optim_epoch))
        torch.save(policy.policy_net.state_dict(), './{}:Policy_model.pkl'.format(args.ppo_optim_epoch))
        torch.save(value.state_dict(), './{}:Value_model.pkl'.format(args.ppo_optim_epoch))

        writer = SummaryWriter()
        writer.add_scalar('d_loss', d_loss, args.training_epochs)
        writer.add_scalar('p_loss', p_loss, args.training_epochs)
        writer.add_scalar('v_loss', v_loss, args.training_epochs)
        memory.clear_memory()

if __name__ == '__main__':
    main()
