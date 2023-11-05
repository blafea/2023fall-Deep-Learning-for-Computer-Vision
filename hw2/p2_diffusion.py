import torch
from utils import beta_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"


# https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
def sample(model, noise, eta=0):
    model.eval()
    x = noise
    step = 1000
    skip = step // 50
    seq = range(0, step, skip)
    with torch.no_grad():
        x = generalized_steps(x, seq, model, beta_scheduler(step), eta=eta)
        x = x[0][-1]
    return (x[0] - x[0].min())/(x[0].max()-x[0].min())


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(1) * i).to(x.device)
            next_t = torch.tensor([j]).to(device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c1 = (
                kwargs.get("eta", 0)
                * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = (at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et).to(
                torch.float32
            )
            xs.append(xt_next.to("cpu"))

    return xs, x0_preds


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(device), beta.to(device)], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def sample_interpolation(model, z1, z2):
    def slerp(z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return (
            torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
            + torch.sin(alpha * theta) / torch.sin(theta) * z2
        )
    def linear(z1, z2, alpha):
        return ((1 - alpha) * z1 + alpha  * z2)
    alpha = torch.arange(0.0, 1.01, 0.1).to(device)
    z_ = []
    # for i in range(alpha.size(0)):
    #     z_.append(slerp(z1, z2, alpha[i]))
    for i in range(alpha.size(0)):
        z_.append(linear(z1, z2, alpha[i]))
    xs = []
    step = 1000
    skip = step // 50
    seq = range(0, step, skip)
    # Hard coded here, modify to your preferences
    with torch.no_grad():
        for i in range(len(z_)):
            img, _ = generalized_steps(z_[i], seq, model, beta_scheduler(step))
            x = (img[-1] + 1.0) / 2.0
            xs.append(x)
    return xs
