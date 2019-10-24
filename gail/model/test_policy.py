import unittest
import torch
from model.policy import Policy
from random import randint

from utils import ModelArgs


class PolicyTestCase(unittest.TestCase):
    def generate_random_sections(self, total_dim):
        # discard a random value
        randint(2, 8)
        sections_len = randint(2, 8)
        sections = [2] * sections_len
        remain_dim = total_dim - sections_len * 2
        if remain_dim <= 0:
            return self.generate_random_sections(total_dim)
        for i in range(sections_len - 1):
            dim = randint(1, remain_dim)
            sections[i] += dim
            remain_dim -= dim
            if remain_dim <= 1:
                break
        sections[sections_len - 1] += (total_dim - sum(sections))
        assert sum(sections) == total_dim
        return sections

    def setUp(self) -> None:
        args = ModelArgs()
        n_discrete_state = args.n_discrete_state
        n_discrete_action = args.n_discrete_action
        self.n_continuous_state = args.n_continuous_state
        self.n_continuous_action = args.n_continuous_action

        self.policy_discrete_state_sections = self.generate_random_sections(n_discrete_state)
        self.policy_discrete_action_sections = self.generate_random_sections(n_discrete_action)
        self.policy = Policy(self.policy_discrete_action_sections, self.policy_discrete_state_sections)

    def test_log_prob(self):
        memory = self.policy.collect_samples(2048)
        batch = memory.sample()
        discrete_state = torch.stack(batch.discrete_state).to(self.policy.device).squeeze_().detach()
        continuous_state = torch.stack(batch.continuous_state).to(self.policy.device).squeeze_().detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).to(self.policy.device).squeeze_().detach()
        continuous_action = torch.stack(batch.continuous_action).to(self.policy.device).squeeze_().detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).to(self.policy.device).squeeze_().detach()
        discrete_action = torch.stack(batch.discrete_action).to(self.policy.device).squeeze_().detach()

        assert discrete_state.size(1) == sum(self.policy_discrete_state_sections)
        assert continuous_state.size(1) == self.n_continuous_state
        assert next_discrete_state.size(1) == sum(self.policy_discrete_state_sections)
        assert continuous_action.size(1) == self.n_continuous_action
        assert next_continuous_state.size(1) == self.n_continuous_state
        assert discrete_action.size(1) == sum(self.policy_discrete_action_sections)

        old_log_prob = torch.stack(batch.old_log_prob).to(self.policy.device).squeeze_().detach()
        mask = torch.Tensor(batch.mask).to(self.policy.device).squeeze_().detach()
        # it should contain 'done'
        assert (mask == 0).any()

        policy_log_prob_new = self.policy.get_policy_net_log_prob(torch.cat((discrete_state, continuous_state), dim=-1),
                                                                  discrete_action, continuous_action)
        transition_log_prob_new = self.policy.get_transition_net_log_prob(
            torch.cat((discrete_state, continuous_state, discrete_action, continuous_action), dim=-1),
            next_discrete_state,
            next_continuous_state)
        new_log_prob = policy_log_prob_new + transition_log_prob_new
        # Due to we are not update policy gradient, old_log_prob and new_log_prob must be equal
        assert torch.allclose(new_log_prob, old_log_prob)


if __name__ == '__main__':
    unittest.main()
