import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, hidden_size, in_size = 1, out_size = 1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = in_size,
                            hidden_size = hidden_size,
                            batch_first = True)
        self.fc = nn.Linear(hidden_size, out_size)
    
    def forward(self, x, h = None):
        out, _ = self.rnn(x, h)
        last_hidden_states = out[:, -1]
        out = self.fc(last_hidden_states)
        return out
    
rnn_model = RNN(hidden_size = 32)
print(rnn_model)

from torchinfo import summary
print('Check Torch Info')
print(summary(rnn_model, input_size = (187,10,1) ))