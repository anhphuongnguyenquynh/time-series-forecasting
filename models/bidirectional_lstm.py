import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first = True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size *2, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])

        return out
    
blstm_model = BLSTM(input_size = 1, hidden_size = 32, num_layers = 20, num_classes=20)
print(blstm_model.state_dict())

from torchinfo import summary
print('Check Torch Info')
print(summary(blstm_model, input_size = (187,10,1) ))