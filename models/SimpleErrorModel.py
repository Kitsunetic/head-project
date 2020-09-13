import torch.nn as nn


class SimpleErrorModel(nn.Module):
    def __init__(self):
        """
        입력을 있는 그대로 yaw, pitch, roll만 출력해줌
        """
        super(SimpleErrorModel, self).__init__()

    def forward(self, x):
        x = x[:, :3]
        return x

    def state_dict(self):
        return None
