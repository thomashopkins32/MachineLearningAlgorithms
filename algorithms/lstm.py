import torch
import torch.nn as nn


class LSTMBlock(nn.Module):
    ''' Long-Short Term Memory Block '''
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.forget_gate = nn.Sequential(nn.Linear(in_features + hidden_size, hidden_size), nn.Sigmoid())
        self.input_gate_sig = nn.Sequential(nn.Linear(in_features + hidden_size, hidden_size), nn.Sigmoid())
        self.input_gate_tanh = nn.Sequential(nn.Linear(in_features + hidden_size, hidden_size), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Linear(in_features + hidden_size, hidden_size), nn.Sigmoid())
        self.in_features = in_features
        self.hidden_size = hidden_size

    def forward(self, x, c, h):
        xx = torch.cat((x, h), dim=1)
        forget_out = self.forget_gate(xx)
        input_out = self.input_gate_sig(xx) * self.input_gate_tanh(xx)
        output_out = self.output_gate(xx)
        c_new = forget_out * c + input_out
        h_new = torch.tanh(c_new) * output_out
        return c_new, h_new


class LSTM(nn.Module):
    ''' Long-Short Term Memory Network '''
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.block = LSTMBlock(in_features, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        '''
        Takes in a batch of sequences and outputs the hidden & cell states over time.
        x.shape should be (bs, t, in_features)
        '''
        hidden = torch.zeros((x.shape[0], self.hidden_size), requires_grad=False)
        cell = torch.zeros((x.shape[0], self.hidden_size), requires_grad=False)
        outs = []
        for i in range(x.shape[1]):
            cell, hidden = self.block(x[:, i, :], cell, hidden)
            outs.append(torch.stack((cell, hidden)))
        return torch.stack(outs)