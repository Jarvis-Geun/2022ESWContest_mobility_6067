import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_features, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        output = self.linear3(x)
        return output

if __name__ == "__main__":
    model = LinearModel(input_features = 9, ourput_features = 1)
    print(model)