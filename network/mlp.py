from torch import nn


class MLP(nn.Module):
    def __init__(self, observation_space, action_space, units_per_layer=24) -> None:
        super().__init__()

        self.action_space = action_space

        self.net = nn.Sequential(
            nn.Linear(observation_space, units_per_layer),
            nn.ReLU(),
            nn.Linear(units_per_layer, units_per_layer),
            nn.ReLU(),
            nn.Linear(units_per_layer, action_space),
        )

    def forward(self, x):
        return self.net(x)

