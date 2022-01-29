import random
import torch
import numpy as np
from pathlib import Path
from collections import namedtuple
from torch import nn, optim


Output = namedtuple("Output", ["loss"])


class DQN:
    def __init__(self, policy_net, target_net, device, hyps):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.hyps = hyps
        self.exploration_rate = hyps["exploration_max"]

    def setup(self):
        self.create_criterion()
        self.create_optimizer()
        self.policy_net.to(self.device)
        if self.target_net is not None:
            self.target_net.to(self.device)

    def create_criterion(self):
        self.criterion = nn.MSELoss()

    def create_optimizer(self):
        self.policy_net_optim = optim.Adam(
            self.policy_net.parameters(), lr=self.hyps["lr"]
        )

    def compute_loss(self, predicted, target):
        return self.criterion(input=predicted, target=target)

    def get_action(self, state, training=True):
        if training and np.random.rand() < self.exploration_rate:
            return random.randrange(self.policy_net.action_space)
        q_values = self.policy_net(torch.FloatTensor(state))
        return np.argmax(q_values.detach()[0].numpy())


    def get_pred_and_tgt(self, state, action, reward, next_state, done):
        with torch.no_grad():
            predicted_best_next_action = (
                self.policy_net(next_state).argmax(dim=1).unsqueeze(-1)
            )
            target_q_value_next_action = self.target_net(next_state).gather(
                dim=1, index=predicted_best_next_action
            )
            target_q_value_curr_action = (
                reward + self.hyps["gamma"] * target_q_value_next_action
            )

        predicted_q_value_curr_action = self.policy_net(state).gather(
            dim=1, index=action
        )
        return predicted_q_value_curr_action, target_q_value_curr_action

    def fit_one_cycle(self, batch):
        self.exploration_rate *= self.hyps["exploration_decay"]
        self.exploration_rate = max(self.exploration_rate, self.hyps["exploration_min"])

        s = torch.FloatTensor(np.array(batch[0]).squeeze())
        a = torch.LongTensor(batch[1]).unsqueeze(-1)
        r = torch.FloatTensor(batch[2]).unsqueeze(-1)
        ns = torch.FloatTensor(np.array(batch[3]).squeeze())
        d = torch.LongTensor(batch[4]).unsqueeze(-1)

        y_pred, y_true = self.get_pred_and_tgt(s, a, r, ns, d)
        batch_loss = self.compute_loss(y_true, y_pred)

        self.policy_net_optim.zero_grad()
        batch_loss.backward()
        self.policy_net_optim.step()

        return Output(loss=batch_loss.item())

    def save(self, model_dir):
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), model_path / "best.pth")

    def load(self, model_dir):
        model_path = Path(model_dir)
        self.policy_net.load_state_dict(torch.load(model_path / "best.pth"))
