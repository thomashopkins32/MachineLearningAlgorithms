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
        self.attention = SelfAttention(kdim)
        self.out = nn.Linear(vdim * nheads, mdim, bias=False)

    def forward(self, q, k, v, mask=None):
        x = self.attention(q, k, v, mask=mask)
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
        o = self.multi_attention1(x, x, x, mask=torch.triu_indices(x.shape[1], x.shape[2]))
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


class PositionalEncoder(nn.Module):
    def __init__(self, mdim, max_length):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        #print(f'position: {position.shape}')
        self.encoder = torch.zeros(1, max_length, mdim)
        #print(f'func: {torch.sin(position / (10000 ** ((2 * torch.arange(0, mdim, 2)) / mdim))).shape}')
        self.encoder[0, :, 0::2] = torch.sin(position / (10000 ** ((2 * torch.arange(0, mdim, 2)) / mdim)))
        self.encoder[0, :, 1::2] = torch.cos(position / (10000 ** ((2 * torch.arange(0, mdim, 2)) / mdim)))

    def forward(self, x):
        return x + self.encoder[:, :x.shape[1], :]


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength):
        super().__init__()
        self.in_embeds = nn.Embedding(vocab_size, mdim)
        self.out_embeds = nn.Embedding(vocab_size, mdim)

        self.pos_encoder = PositionalEncoder(mdim, sqlength)

        self.encoder = TransformerEncoder(n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength)
        self.decoder = TransformerDecoder(n_layers, nheads, mdim, kdim, vdim, ffdim, sqlength)

        self.out = nn.Linear(mdim, vocab_size)

    def forward(self, x):
        inputs = x
        print(f'inputs: {inputs.shape}')
        i_emb = self.in_embeds(inputs).squeeze()
        print(f'i_emb1: {i_emb.shape}')
        i_emb = self.pos_encoder(i_emb).squeeze()
        print(f'i_emb2: {i_emb.shape}')

        outputs = x[:, 1:, :]
        print(f'outputs: {outputs.shape}')
        o_emb = self.out_embeds(outputs).squeeze()
        print(f'o_emb1: {o_emb.shape}')
        o_emb = self.pos_encoder(o_emb).squeeze()
        print(f'o_emb2: {o_emb.shape}')

        enc = self.encoder(i_emb)
        print(f'enc: {enc.shape}')
        x = self.decoder(o_emb, enc)
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
