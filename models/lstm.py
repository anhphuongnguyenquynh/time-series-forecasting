import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, hidden_size, in_size = 1, out_size = 1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = in_size,
            hidden_size = hidden_size,
            batch_first = True)
        self.fc = nn.Linear(hidden_size, out_size)
    
    def forward(self, x, h = None):
        out, h = self.lstm(x, h)
        last_hidden_states = out[:, -1, :]
        out = self.fc(last_hidden_states)
        return out
    
lstm_model = LSTM(hidden_size = 32)
print(lstm_model)

from torchinfo import summary
print('Check Torch Info')
print(summary(lstm_model, input_size = (187,10,1) ))