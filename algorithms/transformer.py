import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, kdim):
        super().__init__()
        self.kdim = kdim

    def forward(self, q, k, v, mask=None):
        '''
        Parameters
        ----------
        q : torch.Tensor
            Query of dim (batch size, sequence length, kdim)
        k : torch.Tensor
            Keys of dim (batch size, sequence length, kdim)
        v : torch.Tensor
            Values of dim (batch size, sequence length, vdim)
        mask : torch.Tensor, optional
            Mask to apply to avoid information leakage from the future
        '''
        x = q @ k.T 
        x = x / torch.sqrt(self.kdim)
        if mask is not None:
            x[mask] = -torch.inf
        x = torch.softmax(x, dim=1)
        return x @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, nheads, mdim, kdim, vdim):
        super().__init__()
        self.nheads = nheads

        self.qs = nn.ModuleList([nn.Linear(mdim, kdim, bias=False) for _ in range(nheads)])
        self.ks = nn.ModuleList([nn.Linear(mdim, kdim, bias=False) for _ in range(nheads)])
        self.vs = nn.ModuleList([nn.Linear(mdim, vdim, bias=False) for _ in range(nheads)])
        self.attention = SelfAttention(kdim)

        self.out = nn.Linear(vdim * nheads, mdim, bias=False)

    def forward(self, q, k, v, mask=None):
        res = []
        # TODO: make run in parallel
        for q, k, v in zip(self.qs, self.ks, self.vs):
            res.append(self.attention(q, k, v, mask=mask))
        x = torch.stack(res, dim=2)
        return self.out(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.multi_attention = MultiHeadAttention(nheads, mdim, kdim, vdim)
        self.ln1 = nn.LayerNorm((sqlength, mdim))

        self.l1 = nn.Linear(mdim, ffdim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(ffdim, mdim)
        self.ln2 = nn.LayerNorm((sqlength, mdim))
    
    def forward(self, x):
        o = self.multi_attention(x, x, x, mask=None)
        x = o + x
        x = self.ln1(x)

        outs = []
        for i in range(x.shape[1]):
            outs.append(self.l2(self.relu(self.l1(x[:, i, :]))))
        o = torch.stack(outs, dim=1)
        x = o + x
        return self.ln2(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.multi_attention1 = MultiHeadAttention(nheads, mdim, kdim, vdim)
        self.ln1 = nn.LayerNorm((sqlength, mdim))
        
        self.multi_attention2 = MultiHeadAttention(nheads, mdim, kdim, vdim)
        self.ln2 = nn.LayerNorm((sqlength, mdim))

        self.l1 = nn.Linear(mdim, ffdim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(ffdim, mdim)
        self.ln2 = nn.LayerNorm((sqlength, mdim))
    
    def forward(self, x, enc):
        # TODO: Implement masking
        o = self.multi_attention1(x, x, x, mask=None)
        x = o + x
        x = self.ln1(x)

        o = self.multi_attention2(enc, enc, x)
        x = o + x
        x = self.ln2(x)

        outs = []
        for i in range(x.shape[1]):
            outs.append(self.l2(self.relu(self.l1(x[:, i, :]))))
        o = torch.stack(outs, dim=1)
        x = o + x
        return self.ln2(x)


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(nheads, mdim, kdim, vdim, ffdim, sqlength)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(nheads, mdim, kdim, vdim, ffdim, sqlength)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, enc):
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, enc)
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.in_embeds = nn.Embedding()
        self.out_embeds = nn.Embedding()

        self.encoder = TransformerEncoder(n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength)
        self.decoder = TransformerDecoder(n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength)

    def forward(self, x):
        # TODO: finish
        inputs = x
        x_emb = self.in_embeds(x)

