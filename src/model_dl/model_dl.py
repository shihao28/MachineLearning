import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_features_count, hidden_layer_sizes, num_classes):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Sequential()
        prev_layer_size = input_features_count
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):
            self.hidden_layer.add_module(
                f"layer{i}", nn.Linear(prev_layer_size, hidden_layer_size))
            self.hidden_layer.add_module(
                f"act{i}", nn.ReLU(inplace=True))
            prev_layer_size = hidden_layer_size
        self.output_layer = nn.Linear(
            prev_layer_size,
            num_classes-1 if num_classes==2 else num_classes)

    def forward(self, x):
        x = self.hidden_layer(x)
        logits = self.output_layer(x)

        return logits


if __name__ == "__main__":
    model = MLP(5, (8, 9, 10), 3)
