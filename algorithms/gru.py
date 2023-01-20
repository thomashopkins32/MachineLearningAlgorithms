import torch
import torch.nn as nn


class GRU(nn.Module):
    ''' Gated Recurrent Unit '''
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Update Gate
        self.Wz = nn.Linear(in_features, hidden_size, bias=False)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        # Reset Gate
        self.Wr = nn.Linear(in_features, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        # Candidate Activation Vector
        self.Wh = nn.Linear(in_features, hidden_size, bias=False)
        self.Uh = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, h):
        z = self.sigmoid(self.Wz(x) + self.Uz(h))
        r = self.sigmoid(self.Wr(x) + self.Ur(h))
        hhat = self.tanh(self.Wh(x) + self.Uh(r * h))
        h_new = z * h + (1 - z) * hhat
        return h_new


class GRUNet(nn.Module):
    ''' Gated Recurrent Network '''
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.block = GRU(in_features, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        '''
        Takes in a batch of sequences and outputs the hidden states over time.
        x.shape should be (bs, t, in_features)
        '''
        hidden = torch.zeros((x.shape[0], self.hidden_size), requires_grad=False)
        outs = []
        for i in range(x.shape[1]):
            hidden = self.block(x[:, i, :], hidden)
            outs.append(hidden)
        return torch.stack(outs).view((x.shape[0], x.shape[1], self.hidden_size))


if __name__ == '__main__':
    test_model = GRUNet(10, 32)
    test_x = torch.randn((32, 5, 10))
    hidden_states = test_model(test_x)
    print(hidden_states.shape)
    print(hidden_states)