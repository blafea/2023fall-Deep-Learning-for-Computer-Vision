import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora


class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.img_embd = 1024
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.peft = "lora"
        self.prefix_num = 128
        self.adapter_size = 128


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer(
            "bias", torch.tril(torch.ones(size, size)).view(1, 1, size, size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, context, embedding
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_qry = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_key = nn.Linear(cfg.img_embd, cfg.n_embd)
        self.c_val = nn.Linear(cfg.img_embd, cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer(
            "bias", torch.tril(torch.ones(size, size)).view(1, 1, size, size)
        )

    def forward(self, qry, key, val):
        B, T, C = qry.size()  # batch, context, embedding
        B2, T2, C2 = key.size()
        q = self.c_qry(qry).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.c_key(key).view(B2, T2, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.c_val(val).view(B2, T2, self.n_head, C // self.n_head).transpose(1, 2)
        # print(v.shape)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        # print(att.shape)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cro_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(
            collections.OrderedDict(
                [
                    ("c_fc", nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
                    ("act", nn.GELU(approximate="tanh")),
                    ("c_proj", nn.Linear(4 * cfg.n_embd, cfg.n_embd)),
                ]
            )
        )
        if cfg.peft == "adapter":
            self.adapter = nn.Sequential(
                collections.OrderedDict(
                    [
                        ("c_fc", nn.Linear(cfg.n_embd, cfg.adapter_size)),
                        ("act", nn.GELU(approximate="tanh")),
                        ("c_proj", nn.Linear(cfg.adapter_size, cfg.n_embd)),
                    ]
                )
            )
        self.dropout = nn.ModuleDict(
            {"cro": nn.Dropout(), "att": nn.Dropout(), "mlp": nn.Dropout()}
        )

    def forward(self, data):
        x, key, val = data
        x = x + self.attn(self.ln_1(x))
        x = x + self.cro_attn(self.ln_3(x), key, val)
        if self.cfg.peft == "adapter":
            x = x + self.mlp(self.adapter(self.ln_2(x)))
        else:
            x = x + self.mlp(self.ln_2(x))
        return (x, key, val)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                h=nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # applying peft
        if cfg.peft == "prefix":
            self.prefix = nn.Embedding(cfg.prefix_num, cfg.n_embd)
        elif cfg.peft == "lora":
            self.transformer.wte = lora.Embedding(cfg.vocab_size, cfg.n_embd, r=4)
            # for name, param in self.transformer.wte.named_parameters():
            #     print(name, param)
            self.transformer.wpe = lora.Embedding(cfg.block_size, cfg.n_embd, r=4)
            self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=4)
            for i in range(cfg.n_layer):
                self.transformer.h[i].attn.c_attn = lora.Linear(
                    cfg.n_embd, 3 * cfg.n_embd, r=4
                )
                self.transformer.h[i].attn.c_proj = lora.Linear(
                    cfg.n_embd, cfg.n_embd, r=4
                )

                self.transformer.h[i].mlp.c_fc = lora.Linear(
                    cfg.n_embd, 4 * cfg.n_embd, r=4
                )
                self.transformer.h[i].mlp.c_proj = lora.Linear(
                    4 * cfg.n_embd, cfg.n_embd, r=4
                )
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [".c_attn.weight", ".c_fc.weight", ".c_proj.weight"]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
                if key.endswith("ln_3.weight"):
                    state_dict[key] = state_dict[
                        key.replace("ln_3.weight", "ln_1.weight")
                    ]
                    state_dict[key] = state_dict[key.replace("ln_3.bias", "ln_1.bias")]

    def forward(self, x: Tensor, key, val):
        x[x == -100] = 50256
        x = x[:, :-1]
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        if self.cfg.peft == "prefix":
            pre = self.prefix(torch.range(0, self.cfg.prefix_num))
            x = torch.cat((pre, x), dim=1)
        x, key, val = self.transformer.h((x, key, val))
        x = self.lm_head(self.transformer.ln_f(x))

        x = torch.swapaxes(x, 1, 2)
        return x
