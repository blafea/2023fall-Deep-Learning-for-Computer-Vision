import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora
import timm


class Config:
    def __init__(self, checkpoint=None, peft=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.img_embd = 1024
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.peft = peft
        self.prefix_num = 100
        self.adapter_size = 64
        self.lora_r = 4


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
        self.att_map = nn.Identity()
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
        att = self.att_map(att)
        # print(att.shape)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cfg = cfg
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
            self.adapter2 = nn.Sequential(
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
        if self.cfg.peft == "adapter":
            x = x + self.adapter2(x)
        x = x + self.cro_attn(self.ln_3(x), key, val)
        if self.cfg.peft == "adapter":
            x = self.ln_2(x)
            x = x + self.adapter(x)
            x = x + self.mlp(x)
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
            self.prefix_pos = nn.Embedding(cfg.prefix_num, cfg.n_embd)
        elif cfg.peft == "lora":
            for i in range(cfg.n_layer):
                self.transformer.h[i].attn.c_attn = lora.Linear(
                    cfg.n_embd, 3 * cfg.n_embd, r=cfg.lora_r
                )
                self.transformer.h[i].attn.c_proj = lora.Linear(
                    cfg.n_embd, cfg.n_embd, r=cfg.lora_r
                )

                self.transformer.h[i].mlp.c_fc = lora.Linear(
                    cfg.n_embd, 4 * cfg.n_embd, r=cfg.lora_r
                )
                self.transformer.h[i].mlp.c_proj = lora.Linear(
                    4 * cfg.n_embd, cfg.n_embd, r=cfg.lora_r
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
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, key, val):
        x[x == -100] = 50256
        x = x[:, :-1]
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        if self.cfg.peft == "prefix":
            pre = torch.arange(0, self.cfg.prefix_num, dtype=torch.long, device="cuda")
            pre = torch.cat([pre for _ in range(x.size()[0])]).reshape(
                x.size()[0], self.cfg.prefix_num
            )
            pos_pre = torch.arange(
                pre.size()[1], dtype=torch.long, device=pre.device
            ).unsqueeze(0)
            pre = self.prefix(pre) + self.prefix_pos(pos_pre)

            x = torch.cat((pre, x), dim=1)
        x, key, val = self.transformer.h((x, key, val))
        x = self.lm_head(self.transformer.ln_f(x))
        if self.cfg.peft == "prefix":
            x = x[:, self.cfg.prefix_num :]

        x = torch.swapaxes(x, 1, 2)
        return x


class Transformer_encdec(nn.Module):
    def __init__(self, enc, dec, peft=None):
        super().__init__()
        self.enc = timm.create_model(enc, pretrained=True)
        self.dec = Decoder(Config(dec, peft))
        self.peft = peft
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, tgt, x):
        x = self.enc.forward_features(x)
        x = self.dec(tgt, x, x)
        return x

    def greedy_search(self, img, max_length=30):
        def forward(memory, x):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.dec.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.dec.transformer.wte(x) + self.dec.transformer.wpe(pos)
            if self.peft == "prefix":
                pre = torch.arange(
                    0, self.dec.cfg.prefix_num, dtype=torch.long, device="cuda"
                )
                pre = torch.cat([pre for _ in range(x.size()[0])]).reshape(
                    x.size()[0], self.dec.cfg.prefix_num
                )
                pos_pre = torch.arange(
                    pre.size()[1], dtype=torch.long, device=pre.device
                ).unsqueeze(0)
                pre = self.dec.prefix(pre) + self.dec.prefix_pos(pos_pre)

                x = torch.cat((pre, x), dim=1)
            x, key, val = self.dec.transformer.h((x, memory, memory))
            x = self.dec.lm_head(self.dec.transformer.ln_f(x)[:, -1])
            return x

        self.eval()
        with torch.no_grad():
            memory = self.enc.forward_features(img)

        curr_state = (
            torch.tensor([50256] * img.shape[0]).reshape((-1, 1)).to(self.device)
        )
        for _ in range(max_length):
            with torch.no_grad():
                output_id = forward(memory, curr_state)
            next_word = output_id.argmax(dim=-1).unsqueeze(1)
            curr_state = torch.concat((curr_state, next_word), dim=1)
        curr_state = curr_state.cpu().numpy()
        preds = []
        for sentence in curr_state:
            count = 0
            for pos in range(len(sentence)):
                if sentence[pos] == 50256:
                    count += 1
                if count == 2:
                    break
            preds.append(sentence[1:pos])

        return preds

    def beam_search(self, img, beams=3, max_length=60):
        def forward(memory, x):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.dec.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.dec.transformer.wte(x) + self.dec.transformer.wpe(pos)
            if self.peft == "prefix":
                pre = torch.arange(
                    0, self.dec.cfg.prefix_num, dtype=torch.long, device="cuda"
                )
                pre = torch.cat([pre for _ in range(x.size()[0])]).reshape(
                    x.size()[0], self.dec.cfg.prefix_num
                )
                pos_pre = torch.arange(
                    pre.size()[1], dtype=torch.long, device=pre.device
                ).unsqueeze(0)
                pre = self.dec.prefix(pre) + self.dec.prefix_pos(pos_pre)

                x = torch.cat((pre, x), dim=1)
            x, key, val = self.dec.transformer.h((x, memory, memory))
            x = self.dec.lm_head(self.dec.transformer.ln_f(x)[:, -1])
            return x

        self.eval()
        with torch.no_grad():
            memory = self.enc.forward_features(img)
        curr_state = torch.tensor([50256]).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_word = forward(memory, curr_state)
        curr_probs, next_chars = next_word.log_softmax(-1).topk(k=beams, axis=-1)
        curr_probs = curr_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)
        curr_state = torch.cat([curr_state] * beams)
        curr_state = torch.cat((curr_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            with torch.no_grad():
                next_probs = forward(
                    torch.cat([memory] * beams), curr_state
                ).log_softmax(-1)
            curr_probs = curr_probs.unsqueeze(-1) + next_probs
            curr_probs = curr_probs.flatten()
            _, idx = curr_probs.topk(k=beams, dim=-1)
            curr_probs = curr_probs[idx]
            next_chars = torch.remainder(idx, self.dec.cfg.vocab_size)
            next_chars = next_chars.unsqueeze(-1)
            top_beams = (idx / self.dec.cfg.vocab_size).long()
            curr_state = curr_state[top_beams]
            curr_state = torch.cat((curr_state, next_chars), dim=1)

            finish_idx = []
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == 50256:
                    ans_ids.append(curr_state[idx].cpu().tolist())
                    ans_probs.append(curr_probs[idx].item() / len(ans_ids[-1]))
                    finish_idx.append(idx)
                    beams -= 1
            keep_idx = [i for i in range(len(curr_state)) if i not in finish_idx]
            if len(keep_idx) == 0:
                break
            curr_state = curr_state[keep_idx]
            curr_probs = curr_probs[keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()
        return ans_ids[max_idx][1:-1]
