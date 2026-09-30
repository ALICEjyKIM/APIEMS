"""GNN 가치 근사 (torch_geometric 없이).
거래 연결 그래프(env/graph.graph)를 입력으로, 종류별 임베딩 → 이웃 평균 메시지 전달 gnn_layers층 → 종류별 평균 풀링 + 활동 참여자 수 → V(s).
주문자·공급자는 같은 열을 공유하지 않으므로 표준화·임베딩·메시지 가중치를 종류별로 따로 두고, 풀링은 자리 수와 무관한 평균을 쓴다.
"""
import numpy as np
import torch
from env.graph import graph, dims, split
from vfa import train


# 은닉 크기 h의 파라미터 수: 종류별 임베딩 2h(d + 1), 층마다 자기·메시지 4h² + 2h, 출력 2h² + 4h + 1
def n_params(cfg, h):
  d = dims(cfg)
  return 2 * h * (d + 1) + cfg.gnn_layers * (4 * h * h + 2 * h) + 2 * h * h + 4 * h + 1


# MLP 파라미터 수 (시장 요약 지표 입력, 은닉 nn_layers층 × nn_hidden)
def mlp_params(cfg):
  f, h = 2 + 5 * cfg.n_items + 2, cfg.nn_hidden
  return f * h + h + (cfg.nn_layers - 1) * (h * h + h) + h + 1


# 은닉 크기: gnn_hidden이 0이면 MLP 파라미터 수에 가장 가까운 값을 쓴다 (같은 튜닝 예산·비슷한 용량)
def hidden(cfg):
  return cfg.gnn_hidden or min(range(4, 512), key=lambda h: abs(n_params(cfg, h) - mlp_params(cfg)))


class Net(torch.nn.Module):
  """주문자·공급자 이분 그래프 신경망.
  종류별 임베딩 뒤, 층마다 자기 상태와 이웃(연결선) 평균을 종류별 가중치로 합쳐 갱신하고 비활성 자리는 0으로 둔다.
  종류별 평균 풀링과 표준화한 활동 참여자 수를 이어 붙여 출력층으로 가치를 낸다 (자리 수에 무관).
  """

  def __init__(self, d, h, layers):
    super().__init__()
    self.eb, self.es = torch.nn.Linear(d, h), torch.nn.Linear(d, h)
    self.wb = torch.nn.ModuleList(torch.nn.Linear(h, h) for _ in range(layers))
    self.ws = torch.nn.ModuleList(torch.nn.Linear(h, h) for _ in range(layers))
    self.gb = torch.nn.ModuleList(torch.nn.Linear(h, h, bias=False) for _ in range(layers))
    self.gs = torch.nn.ModuleList(torch.nn.Linear(h, h, bias=False) for _ in range(layers))
    self.out = torch.nn.Sequential(torch.nn.Linear(2 * h + 2, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

  # 순전파: xb·xs (N × 자리 × 특징), kb·ks (N × 자리) 활성 마스크, A (N × 주문자 자리 × 공급자 자리), cnt (N × 2) 표준화한 참여자 수
  def forward(self, xb, xs, kb, ks, A, cnt):
    ub, us = kb[..., None], ks[..., None]
    hb, hs = torch.relu(self.eb(xb)) * ub, torch.relu(self.es(xs)) * us
    db, ds = A.sum(2, keepdim=True).clamp(min=1), A.sum(1)[..., None].clamp(min=1)
    for wb, ws, gb, gs in zip(self.wb, self.ws, self.gb, self.gs):
      ab, as_ = A @ hs / db, A.transpose(1, 2) @ hb / ds
      hb, hs = torch.relu(wb(hb) + gb(ab)) * ub, torch.relu(ws(hs) + gs(as_)) * us
    pb = hb.sum(1) / kb.sum(1, keepdim=True).clamp(min=1)
    ps = hs.sum(1) / ks.sum(1, keepdim=True).clamp(min=1)
    return self.out(torch.cat([pb, ps, cnt], 1))[:, 0]


class GNN:
  """GNN 가치 근사 (fit/predict는 Linear·MLP와 같은 인터페이스, 입력은 graph()로 편 배열).
  fit은 활성 노드를 종류별로 모아 특징을, 학습 데이터로 참여자 수와 목표를 표준화하고 Adam(weight decay wd)으로 전체 배치 nn_epochs회 학습한다.
  enc는 관측 → 입력 배열 변환이며 유지 가치 계산(vfa/coef.raw)이 쓴다.
  """

  def __init__(self, cfg, wd):
    self.cfg, self.wd = cfg, wd
    self.enc = lambda o: graph(cfg, o)

  # 편 배열 → 표준화한 텐서 목록
  def _unpack(self, X):
    xb, xs, kb, ks, A = split(self.cfg, X)
    xb = (xb - self.mu_b) / self.sd_b * kb[..., None]
    xs = (xs - self.mu_s) / self.sd_s * ks[..., None]
    cnt = (np.c_[kb.sum(1), ks.sum(1)] - self.mu_n) / self.sd_n
    return [torch.tensor(a, dtype=torch.float32) for a in (xb, xs, kb, ks, A, cnt)]

  # 학습: 종류별 표준화 후 평균제곱오차를 전체 배치로 최소화
  def fit(self, X, y):
    c = self.cfg
    torch.set_num_threads(1)
    torch.manual_seed(c.nn_seed)
    xb, xs, kb, ks, _ = split(c, X)
    ab, as_ = xb[kb.astype(bool)], xs[ks.astype(bool)]
    std = lambda a: (a.mean(0), np.where(a.std(0) > 0, a.std(0), 1.0))
    self.mu_b, self.sd_b = std(ab) if len(ab) else (np.zeros(xb.shape[2]), np.ones(xb.shape[2]))
    self.mu_s, self.sd_s = std(as_) if len(as_) else (np.zeros(xs.shape[2]), np.ones(xs.shape[2]))
    self.mu_n, self.sd_n = std(np.c_[kb.sum(1), ks.sum(1)])
    self.ym, self.ys = y.mean(), y.std() or 1.0
    self.h = hidden(c)
    self.net = Net(dims(c), self.h, c.gnn_layers)
    xs_ = self._unpack(X)
    t = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
    opt = torch.optim.Adam(self.net.parameters(), lr=c.nn_lr, weight_decay=self.wd)
    for _ in range(c.nn_epochs):
      opt.zero_grad()
      ((self.net(*xs_) - t) ** 2).mean().backward()
      opt.step()
    return self

  # 가치 예측
  def predict(self, X):
    with torch.no_grad():
      return self.net(*self._unpack(X)).numpy().astype(float) * self.ys + self.ym


# weight decay 선택: nn_wds 중 공통 검증 분할의 평균제곱오차가 가장 작은 값으로 전체 데이터를 다시 맞춘다 (MLP와 같은 예산)
def select(cfg, X, y, g):
  return train.select(cfg, lambda wd: GNN(cfg, wd), cfg.nn_wds, X, y, g)
