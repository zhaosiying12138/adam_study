
import math
import torch

torch.set_printoptions(precision=10)

w = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
optimizer = torch.optim.Adam(
    [w],
    lr=0.1,
    betas=(0.9, 0.999),
    eps=1e-8,
    foreach=False,
    fused=False,
)

beta1, beta2 = optimizer.param_groups[0]["betas"]
lr = optimizer.param_groups[0]["lr"]
eps = optimizer.param_groups[0]["eps"]

manual_m = 0.0
manual_v = 0.0

print(f"torch version: {torch.__version__}")
print("Demo loss: L(w) = 0.5 * (w - 4)^2, so grad = (w - 4)")
print(f"initial w = {w.item():.10f}\n")

for step in range(1, 6):
    optimizer.zero_grad()
    loss = 0.5 * (w - 4.0) ** 2
    loss.backward()
    grad = w.grad.item()

    manual_m = beta1 * manual_m + (1 - beta1) * grad
    manual_v = beta2 * manual_v + (1 - beta2) * (grad ** 2)
    m_hat = manual_m / (1 - beta1 ** step)
    v_hat = manual_v / (1 - beta2 ** step)
    delta = -lr * m_hat / (math.sqrt(v_hat) + eps)
    manual_next_w = w.item() + delta

    w_before = w.item()
    optimizer.step()
    state = optimizer.state[w]
    exp_avg = float(state["exp_avg"].item())
    exp_avg_sq = float(state["exp_avg_sq"].item())
    w_after = w.item()

    print(f"step {step}")
    print(f"  w_before         = {w_before:.10f}")
    print(f"  grad             = {grad:.10f}")
    print(f"  exp_avg (m_t)    = {exp_avg:.10f}")
    print(f"  exp_avg_sq (v_t) = {exp_avg_sq:.10f}")
    print(f"  m_hat            = {m_hat:.10f}")
    print(f"  v_hat            = {v_hat:.10f}")
    print(f"  update delta     = {delta:.10f}")
    print(f"  w_after          = {w_after:.10f}")
    print(f"  manual_next_w    = {manual_next_w:.10f}")
    print(f"  match?           = {abs(w_after - manual_next_w) < 1e-12}\n")

