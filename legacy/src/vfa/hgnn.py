"""Buyer–Item–Supplier heterogeneous SUM-GNN (공급자 노드별 c_i 직접 출력).
상태 하나를 주문자·품목·공급자 노드와 주문자–품목(요구수량, 마크업), 공급자–품목(공급가능량, 공급가격 수준) 연결선으로 받는다.
층 k가 짝수면 주문자·공급자 → 품목, 홀수면 품목 → 주문자·공급자로 SUM 메시지를 보내고, 공급자마다 [자기 표현, Σ 연결 품목 메시지]에서 c_i를 낸다.
"""
import numpy as np
import torch
from torch import nn

KEYS = ("xb", "xi", "xs", "ebi", "esi")


# 그래프 묶음: 노드·연결선 배열을 이어 붙이고 연결선 끝 번호에 노드 offset을 더한다
def batch(graphs):
  out = {k: np.concatenate([g[k] for g in graphs]) for k in KEYS}
  ob = oi = os_ = 0
  bi_b, bi_i, si_s, si_i = [], [], [], []
  for g in graphs:
    bi_b.append(g["bi_b"] + ob)
    bi_i.append(g["bi_i"] + oi)
    si_s.append(g["si_s"] + os_)
    si_i.append(g["si_i"] + oi)
    ob, oi, os_ = ob + len(g["xb"]), oi + len(g["xi"]), os_ + len(g["xs"])
  out.update(bi_b=np.concatenate(bi_b), bi_i=np.concatenate(bi_i), si_s=np.concatenate(si_s), si_i=np.concatenate(si_i))
  return out


class Net(nn.Module):
  """SUM-GNN 본체.
  enc_*: 종류별 입력 → hidden, msg_*: 층별·관계별 메시지 ([보내는 노드, 연결선 특징] → hidden), self_*: 자기 갱신 (쓰는 층에만 만든다).
  read: 공급자 [표현, Σ 연결 품목의 읽기 메시지] → c_i.
  """

  def __init__(self, dims, hid, layers):
    super().__init__()
    db, di, ds, de = dims
    lin = lambda a: nn.Sequential(nn.Linear(a, hid), nn.ReLU())
    self.layers = layers
    self.enc_b, self.enc_i, self.enc_s = lin(db), lin(di), lin(ds)
    even, odd = [str(k) for k in range(0, layers, 2)], [str(k) for k in range(1, layers, 2)]
    md = lambda ks, f: nn.ModuleDict({k: f() for k in ks})
    self.msg_bi, self.msg_si = md(even, lambda: lin(hid + de)), md(even, lambda: lin(hid + de))
    self.msg_ib, self.msg_is = md(odd, lambda: lin(hid + de)), md(odd, lambda: lin(hid + de))
    self.self_i = md(even, lambda: nn.Linear(hid, hid))
    self.self_b, self.self_s = md(odd, lambda: nn.Linear(hid, hid)), md(odd, lambda: nn.Linear(hid, hid))
    self.msg_rd = lin(hid + de)
    self.read = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 1))

  # 노드 n개에 연결선 메시지를 SUM
  @staticmethod
  def agg(n, idx, m):
    return torch.zeros(n, m.shape[1]).index_add_(0, idx, m)

  def forward(self, G):
    hb, hi, hs = self.enc_b(G["xb"]), self.enc_i(G["xi"]), self.enc_s(G["xs"])
    for j in range(self.layers):
      k = str(j)
      if j % 2 == 0:
        m = self.agg(len(hi), G["bi_i"], self.msg_bi[k](torch.cat([hb[G["bi_b"]], G["ebi"]], 1)))
        m = m + self.agg(len(hi), G["si_i"], self.msg_si[k](torch.cat([hs[G["si_s"]], G["esi"]], 1)))
        hi = torch.relu(self.self_i[k](hi) + m)
      else:
        hb = torch.relu(self.self_b[k](hb) + self.agg(len(hb), G["bi_b"], self.msg_ib[k](torch.cat([hi[G["bi_i"]], G["ebi"]], 1))))
        hs = torch.relu(self.self_s[k](hs) + self.agg(len(hs), G["si_s"], self.msg_is[k](torch.cat([hi[G["si_i"]], G["esi"]], 1))))
    r = self.agg(len(hs), G["si_s"], self.msg_rd(torch.cat([hi[G["si_i"]], G["esi"]], 1)))
    return self.read(torch.cat([hs, r], 1))[:, 0]


class HGNN:
  """학습·예측 감싸개.
  fit은 노드·연결선 특징을 학습 그래프 기준으로 표준화하고 목표 c를 표준화해 MSE를 전체 배치 Adam(lr, weight decay wd)으로 최소화한다.
  검증 그래프를 주면 every에폭마다 검증 MSE를 재고 최소였던 에폭의 가중치로 되돌린다 (MLP와 같은 조기 종료 규칙).
  """

  def __init__(self, hid, layers, wd, seed, epochs=1000, every=25, lr=1e-3):
    self.hid, self.layers, self.wd, self.seed, self.epochs, self.every, self.lr = hid, layers, wd, seed, epochs, every, lr

  # numpy 그래프 묶음 → 표준화한 torch 텐서
  def tensors(self, B):
    G = {k: torch.tensor((B[k] - self.mu[k]) / self.sd[k], dtype=torch.float32) for k in KEYS}
    G.update({k: torch.tensor(B[k], dtype=torch.long) for k in ("bi_b", "bi_i", "si_s", "si_i")})
    return G

  def fit(self, graphs, ys, val=None, yv=None):
    torch.set_num_threads(1)
    torch.manual_seed(self.seed)
    B = batch(graphs)
    self.mu = {k: B[k].mean(0) for k in KEYS}
    self.sd = {k: np.where(B[k].std(0) > 0, B[k].std(0), 1.0) for k in KEYS}
    y = np.concatenate(ys)
    self.ym, self.ys = y.mean(), y.std() or 1.0
    dims = (B["xb"].shape[1], B["xi"].shape[1], B["xs"].shape[1], B["ebi"].shape[1])
    assert B["ebi"].shape[1] == B["esi"].shape[1]
    self.net = Net(dims, self.hid, self.layers)
    G, t = self.tensors(B), torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
    opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
    best, state, self.best_epoch = np.inf, None, 0
    if val is not None:
      Gv, yv_ = self.tensors(batch(val)), np.concatenate(yv)
    for e in range(self.epochs):
      opt.zero_grad()
      ((self.net(G) - t) ** 2).mean().backward()
      opt.step()
      if val is not None and (e + 1) % self.every == 0:
        with torch.no_grad():
          mse = float(np.mean((self.net(Gv).numpy() * self.ys + self.ym - yv_) ** 2))
        if mse < best:
          best, self.best_epoch = mse, e + 1
          state = {k: v.clone() for k, v in self.net.state_dict().items()}
    if state is not None:
      self.net.load_state_dict(state)
    return self

  # 그래프 목록 → 그래프별 공급자 c 예측 배열
  def predict(self, graphs):
    with torch.no_grad():
      p = self.net(self.tensors(batch(graphs))).numpy().astype(float) * self.ys + self.ym
    n = np.cumsum([0] + [len(g["xs"]) for g in graphs])
    return [p[a:b] for a, b in zip(n[:-1], n[1:])]

  # 파라미터 수
  def n_params(self):
    return sum(p.numel() for p in self.net.parameters())
