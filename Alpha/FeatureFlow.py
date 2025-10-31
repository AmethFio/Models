import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# 子网络：生成 s 和 t，用于仿射变换
# -----------------------------------------------------
class STNet(nn.Module):
    def __init__(self, in_dim, cond_dim, hid_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim * 2)  # 输出 s 和 t
        )

    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)
        st = self.net(h)
        s, t = torch.chunk(st, 2, dim=1)
        return s, t


# -----------------------------------------------------
# 仿射耦合层（Affine Coupling）
# -----------------------------------------------------
class AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, swap=False):
        super().__init__()
        self.swap = swap
        self.net = STNet(dim // 2, cond_dim)

    def forward(self, x, cond, reverse=False):
        # 可能交换输入
        if self.swap:
            x1, x2 = x.chunk(2, dim=1)
            x1, x2 = x2, x1
        else:
            x1, x2 = x.chunk(2, dim=1)

        s, t = self.net(x1, cond)

        if not reverse:
            # 正向：用于训练
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            # 反向：用于生成
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)

        y = torch.cat([x1, y2], dim=1)

        # 如果交换过，再还原顺序
        if self.swap:
            y1, y2 = y.chunk(2, dim=1)
            y = torch.cat([y2, y1], dim=1)

        return y, log_det


# -----------------------------------------------------
# 条件 Flow 模型：多层耦合结构
# -----------------------------------------------------

class ConditionalFlow(nn.Module):
    def __init__(self, dim, cond_dim, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(dim, cond_dim, swap=(i % 2 == 1))  # 每层交替交换
            for i in range(n_layers)
        ])

    def forward(self, x, cond, reverse=False):
        log_det_total = 0
        if not reverse:
            for layer in self.layers:
                x, log_det = layer(x, cond, reverse=False)
                log_det_total += log_det
        else:
            for layer in reversed(self.layers):
                x, log_det = layer(x, cond, reverse=True)
                log_det_total += log_det
        return x, log_det_total

# === 训练示例 ===
def train_step(flow, x, cond, optimizer):
    z, log_det = flow(x, cond)
    log_prob_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * torch.log(torch.tensor(2 * torch.pi))
    loss = -(log_prob_z + log_det).mean()  # 负对数似然
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def generate(flow, cond, n_samples=4):
    z = torch.randn(n_samples, flow.layers[0].dim).to(cond.device)
    with torch.no_grad():
        x_gen = flow.inverse(z, cond)
    return x_gen


# ======================
# 4️⃣ 生成阶段（z → x）
# ======================
def generate(flow, cond, n_samples=4):
    z = torch.randn(n_samples, flow.layers[0].dim).to(cond.device)
    with torch.no_grad():
        x_gen = flow.inverse(z, cond)
    return x_gen


# ======================
# 5️⃣ 使用示例
# ======================
if __name__ == "__main__":
    dim_x = 16      # 例如 flatten 后的深度特征维度
    dim_cond = 8    # CSI 提取后的特征维度
    flow = ConditionalRealNVP(dim_x, dim_cond).cuda()

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

    for step in range(1000):
        x = torch.randn(32, dim_x).cuda()      # 模拟深度图特征
        cond = torch.randn(32, dim_cond).cuda()  # 模拟CSI特征
        loss = train_step(flow, x, cond, optimizer)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # 生成阶段：给定条件，从噪声生成样本
    cond_test = torch.randn(4, dim_cond).cuda()
    x_gen = generate(flow, cond_test)
    print("Generated samples:", x_gen.shape)