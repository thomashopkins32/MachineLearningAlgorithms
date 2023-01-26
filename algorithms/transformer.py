import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, mdim, kdim, vdim):
        super().__init__()
        self.kdim = kdim
        self.q_proj = nn.Linear(mdim, kdim, bias=False)
        self.k_proj = nn.Linear(mdim, kdim, bias=False)
        self.v_proj = nn.Linear(mdim, vdim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(kdim, kdim))) 

    def forward(self, q, k, v):
        '''
        Parameters
        ----------
        q : torch.Tensor
            Query of dim (batch size, sequence length, kdim)
        k : torch.Tensor
            Keys of dim (batch size, sequence length, kdim)
        v : torch.Tensor
            Values of dim (batch size, sequence length, vdim)
        '''
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        x = q @ k.transpose(-2, -1)
        x = x * self.kdim ** -0.5
        x.masked_fill(self.tril[:q.shape[1], :q.shape[1]] == 0, -torch.inf)
        x = torch.softmax(x, dim=1)
        return x @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, nheads, mdim, kdim, vdim):
        super().__init__()
        self.nheads = nheads
        self.attention = nn.ModuleList((SelfAttention(mdim, kdim, vdim) for _ in range(nheads)))
        self.out = nn.Linear(vdim * nheads, mdim, bias=False)

    def forward(self, q, k, v):
        outs = []
        for n in range(self.nheads):
            outs.append(self.attention[n](q, k, v))
        x = torch.stack(outs, dim=1)
        return self.out(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.multi_attention1 = MultiHeadAttention(nheads, mdim, kdim, vdim)
        self.ln1 = nn.LayerNorm((sqlength, mdim))

        self.l1 = nn.Linear(mdim, ffdim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(ffdim, mdim)
        self.ln2 = nn.LayerNorm((sqlength, mdim))
    
    def forward(self, x):
        o = self.multi_attention1(x, x, x)
        x = o + x
        x = self.ln1(x)

        o = self.l2(self.relu(self.l1(x)))
        x = o + x
        return self.ln2(x)


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(nheads, mdim, kdim, vdim, ffdim, sqlength)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for i in range(len(self.decoders)):
            x = self.decoders[i](x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, mdim, max_length):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        self.encoder = torch.zeros(1, max_length, mdim)
        self.encoder[0, :, 0::2] = torch.sin(position / (10000 ** ((2 * torch.arange(0, mdim, 2)) / mdim)))
        self.encoder[0, :, 1::2] = torch.cos(position / (10000 ** ((2 * torch.arange(0, mdim, 2)) / mdim)))

    def forward(self, x):
        return x + self.encoder[:, :x.shape[1], :]


class Transformer(nn.Module):
    ''' Decoder only since I am not doing translation '''
    def __init__(self, vocab_size, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.in_embeds = nn.Embedding(vocab_size, mdim)

        self.pos_encoder = PositionalEncoder(mdim, sqlength)

        self.decoder = TransformerDecoder(n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength)

        self.out = nn.Linear(mdim, vocab_size)

    def forward(self, x):
        inputs = x
        print(f'inputs: {inputs.shape}')
        i_emb = self.in_embeds(inputs).squeeze()
        print(f'i_emb1: {i_emb.shape}')
        i_emb = self.pos_encoder(i_emb).squeeze()
        print(f'i_emb2: {i_emb.shape}')
        x = self.decoder(i_emb)
        print(f'x: {x.shape}')
        return self.out(x)

    
if __name__ == '__main__':
    vocab_size = 10
    n_layers = 4
    nheads = 4
    mdim = 32
    kdim = 16
    vdim = 8
    ffdim = 64
    sqlength = 5
    model = Transformer(vocab_size, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength)
    x = torch.randint(low=0, high=vocab_size-1, size=(3, sqlength, 1))
    logits = model(x)
    print(logits.shape)
    print(logits)
