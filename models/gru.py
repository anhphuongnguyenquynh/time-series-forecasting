import torch.nn as nn
class GRU(nn.Module):
    def __init__(self, hidden_size, in_size = 1, out_size = 1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size = in_size, hidden_size = hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h = None):
        out, _ = self.gru(x, h)
        last_hidden_states = out[:,-1]
        out = self.fc(last_hidden_states)
        return out  
    
gru_model = GRU(hidden_size = 32)
print(gru_model.state_dict())

from torchinfo import summary
print('Check Torch Info')
print(summary(gru_model, input_size = (187,10,1) ))