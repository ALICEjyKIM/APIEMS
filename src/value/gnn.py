"""GNN Q̂ (질문 3, ref/decisions.md "GNN 구조"): 주문자·품목·공급자 노드와 주문자–품목·공급자–품목·품목–품목 엣지의 이종 그래프 (value.graph).
층마다 엣지 종류별 메시지 φ_r([h_u; e_uv])를 합으로 모으고 노드 종류별 MLP로 갱신한다 (R-GCN식 종류별 가중치, MPNN식 엣지 피처, GINE식 합).
출력은 노드 종류별 합·평균을 표 입력(value.feat.feats)과 붙여 MLP로 후보별 Â를 낸다. 에폭·배치 행 수·조기 종료·난수는 value.fit.mlp와 같다.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from value.graph import DIM

TYPES = ("b", "i", "s")
RELS = (("bi", "bi", "b", "i"), ("ib", "bi", "i", "b"), ("si", "si", "s", "i"), ("is", "si", "i", "s"), ("ii", "ii", "i", "i"))
F = torch.float32


# 표준화 계수: 노드 종류별·엣지 종류별 평균과 표준편차 (0이면 1)
def stats(gs):
  st = {}
  for k in DIM:
    x = np.concatenate([g[k] if k in TYPES else g[k][2] for g in gs])
    sd = x.std(0)
    st[k] = (x.mean(0), np.where(sd > 0, sd, 1.0))
  return st


# 그래프 목록을 한 묶음으로: 노드 행렬과 그래프 번호, 엣지 번호에 노드 offset을 더하고 피처를 표준화
def batch(gs, st):
  B, off = {"n": len(gs)}, {}
  for t in TYPES:
    n = np.array([len(g[t]) for g in gs])
    off[t] = np.r_[0, np.cumsum(n)[:-1]]
    B[t] = torch.tensor((np.concatenate([g[t] for g in gs]) - st[t][0]) / st[t][1], dtype=F)
    B["g" + t] = torch.tensor(np.repeat(np.arange(len(gs)), n))
  for k in ("bi", "si", "ii"):
    a = np.concatenate([g[k][0] + off[k[0]][j] for j, g in enumerate(gs)]).astype(np.int64)
    b = np.concatenate([g[k][1] + off[k[1]][j] for j, g in enumerate(gs)]).astype(np.int64)
    e = (np.concatenate([g[k][2] for g in gs]) - st[k][0]) / st[k][1]
    B[k] = (torch.tensor(a), torch.tensor(b), torch.tensor(e, dtype=F))
  return B


# 행 r(상태 r // k, 후보 r % k)이 쓰는 그래프 번호: 상태마다 m_grid 순 nm개, 후보 j는 m 번째 j // nt (cands가 m 바깥, θ 안쪽)
def rows_graph(n, k, nm):
  r = np.arange(n * k)
  return torch.tensor((r // k) * nm + (r % k) // (k // nm))


class Net(nn.Module):
  """이종 GINE: 노드 종류별 입력층, 층마다 엣지 종류별 메시지 Linear([h_src; e]) → ReLU를 도착 노드에 합하고 노드 종류별 2층 MLP로 갱신.
  출력: 그래프마다 노드 종류별 합·평균(6H)을 그 그래프를 쓰는 후보 행의 표 입력과 붙여 2층 MLP로 값 하나.
  역방향 엣지(ib, is)는 저장된 bi, si의 출발·도착을 바꿔 쓴다.
  """

  def __init__(self, dx, L, H):
    super().__init__()
    self.L = L
    self.enc = nn.ModuleDict({t: nn.Linear(DIM[t], H) for t in TYPES})
    self.msg = nn.ModuleList([nn.ModuleDict({r: nn.Linear(H + DIM[k], H) for r, k, _, _ in RELS}) for _ in range(L)])
    self.upd = nn.ModuleList([nn.ModuleDict({t: nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H)) for t in TYPES}) for _ in range(L)])
    self.out = nn.Sequential(nn.Linear(6 * H + dx, H), nn.ReLU(), nn.Linear(H, 1))

  # 묶음 B, 표 입력 X(행), 행별 그래프 번호 gi → 행별 값
  def forward(self, B, X, gi):
    h = {t: torch.relu(self.enc[t](B[t])) for t in TYPES}
    for l in range(self.L):
      agg = {t: torch.zeros_like(h[t]) for t in TYPES}
      for r, k, s, d in RELS:
        a, b, e = B[k]
        src, dst = (a, b) if s == k[0] else (b, a)
        agg[d] = agg[d].index_add(0, dst, torch.relu(self.msg[l][r](torch.cat([h[s][src], e], 1))))
      h = {t: torch.relu(self.upd[l][t](h[t] + agg[t])) for t in TYPES}
    z = []
    for t in TYPES:
      s = torch.zeros(B["n"], h[t].shape[1]).index_add(0, B["g" + t], h[t])
      c = torch.zeros(B["n"]).index_add(0, B["g" + t], torch.ones(len(h[t])))
      z += [s, s / c.clamp(min=1)[:, None]]
    return self.out(torch.cat([torch.cat(z, 1)[gi], X], 1))[:, 0]


class G:
  """학습한 GNN Q̂: 표 입력 표준화(mu, sd), 목표 배율 ys, 그래프 표준화 st와 net을 담아 (표 입력, 그래프 목록) → 후보별 Â를 낸다.
  여러 상태를 한 번에 넣을 수 있다 (표 입력은 상태마다 후보 k행, 그래프는 상태마다 m_grid 순 nm개). 메모리를 위해 상태 chunk개씩 나눠 계산한다.
  value.fit.Q처럼 Pool로 넘길 수 있게 객체로 둔다.
  """

  def __init__(self, mu, sd, ys, st, net, k, nm, chunk=256):
    self.mu, self.sd, self.ys, self.st, self.net, self.k, self.nm, self.chunk = mu, sd, ys, st, net, k, nm, chunk

  # (표 입력 행렬, 그래프 목록) → Â
  def __call__(self, inp):
    X, gs = inp
    Z, n, out = (np.asarray(X, float) - self.mu) / self.sd, len(X) // self.k, []
    with torch.no_grad():
      for i in range(0, n, self.chunk):
        j = min(i + self.chunk, n)
        out.append(self.net(batch(gs[i * self.nm:j * self.nm], self.st), torch.tensor(Z[i * self.k:j * self.k], dtype=F), rows_graph(j - i, self.k, self.nm)).numpy())
    return np.concatenate(out) * self.ys


# GNN 학습: 층 L × 숨은 H, Adam(lr), 에폭 cfg.mlp_epochs, 한 번에 상태 cfg.mlp_batch // k개(MLP 배치와 같은 행 수), 난수 cfg.seed.
# X: 상태마다 후보 k행의 표 입력, gs: 상태마다 nm개 그래프, Y: 행별 목표. hold = (X, gs, Y)를 주면 10에폭마다 그 제곱오차가 가장 작은 시점을 쓴다
def gnn(cfg, X, gs, Y, mu, sd, ys, k, L, H, lr, hold=None):
  torch.manual_seed(cfg.seed)
  nm, n = len(gs) // (len(X) // k), len(X) // k
  q = G(mu, sd, ys, stats(gs), Net(X.shape[1], L, H), k, nm)
  opt = torch.optim.Adam(q.net.parameters(), lr=lr)
  Z, Yt, g = torch.tensor((X - mu) / sd, dtype=F), torch.tensor(Y / ys, dtype=F), torch.Generator().manual_seed(cfg.seed)
  best = (np.inf, None)
  for e in range(cfg.mlp_epochs):
    for idx in torch.randperm(n, generator=g).split(max(1, cfg.mlp_batch // k)):
      idx = idx.numpy()
      r = (idx[:, None] * k + np.arange(k)).ravel()
      B = batch([gs[i * nm + j] for i in idx for j in range(nm)], q.st)
      opt.zero_grad()
      ((q.net(B, Z[r], rows_graph(len(idx), k, nm)) - Yt[r]) ** 2).mean().backward()
      opt.step()
    if hold is not None and (e + 1) % 10 == 0:
      err = float(((q(hold[:2]) - hold[2]) ** 2).mean())
      best = min(best, (err, deepcopy(q.net.state_dict())), key=lambda b: b[0])
  if best[1] is not None:
    q.net.load_state_dict(best[1])
  q.net.eval()
  return q
