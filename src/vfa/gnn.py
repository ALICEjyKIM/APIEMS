"""GNN 가치 근사 (torch_geometric 없이).
거래 연결 그래프(env/graph.graph)를 입력으로, 노드 임베딩 → 이웃 평균 메시지 전달 gnn_layers층 → 종류별 합 풀링 → 출력으로 V(s)를 맞춘다.
학습량(Adam, 전체 배치 nn_epochs회)과 튜닝 격자(weight decay nn_wds)는 MLP와 같고, 은닉 크기 gnn_hidden은 파라미터 수가 MLP와 비슷하게 둔다.
"""
import numpy as np
import torch
from env.graph import graph
from vfa import train


class Net(torch.nn.Module):
  """주문자·공급자 이분 그래프 신경망.
  입력 임베딩 뒤, 층마다 자기 상태와 이웃(연결선) 평균을 합쳐 갱신하고 비활성 자리는 0으로 둔다.
  주문자 노드 합과 공급자 노드 합을 이어 붙여 출력층으로 가치를 낸다.
  """

  def __init__(self, d, h, layers):
    super().__init__()
    self.emb = torch.nn.Linear(d, h)
    self.self_ = torch.nn.ModuleList(torch.nn.Linear(h, h) for _ in range(layers))
    self.msg = torch.nn.ModuleList(torch.nn.Linear(h, h, bias=False) for _ in range(layers))
    self.out = torch.nn.Sequential(torch.nn.Linear(2 * h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

  # 순전파: x (N × 노드 × 특징), mask (N × 노드), A (N × 주문자 자리 × 공급자 자리)
  def forward(self, x, mask, A):
    nb, m = A.shape[1], mask[..., None]
    h = torch.relu(self.emb(x)) * m
    db, ds = A.sum(2, keepdim=True).clamp(min=1), A.sum(1)[..., None].clamp(min=1)
    for ws, wm in zip(self.self_, self.msg):
      agg = torch.cat([A @ h[:, nb:] / db, A.transpose(1, 2) @ h[:, :nb] / ds], 1)
      h = torch.relu(ws(h) + wm(agg)) * m
    return self.out(torch.cat([h[:, :nb].sum(1), h[:, nb:].sum(1)], 1))[:, 0]


class GNN:
  """GNN 가치 근사 (fit/predict는 Linear·MLP와 같은 인터페이스, 입력은 graph()로 편 배열).
  fit은 활성 노드 기준으로 노드 특징을, 학습 데이터로 목표를 표준화하고 Adam(weight decay wd)으로 전체 배치 nn_epochs회 학습한다.
  enc는 관측 → 입력 배열 변환이며 유지 가치 계산(vfa/coef.raw)이 쓴다.
  """

  def __init__(self, cfg, wd):
    self.cfg, self.wd = cfg, wd
    self.enc = lambda o: graph(cfg, o)

  # 편 배열 → (노드 특징, 마스크, 연결선) 텐서
  def _unpack(self, X):
    c = self.cfg
    n, d, nb = c.gnn_nb + c.n_sup, 2 * c.n_items + 2, c.gnn_nb
    x = X[:, :n * d].reshape(-1, n, d)
    mask = X[:, n * d:n * d + n]
    A = X[:, n * d + n:].reshape(-1, nb, c.n_sup)
    x = (x - self.mu) / self.sd * mask[..., None]
    return [torch.tensor(a, dtype=torch.float32) for a in (x, mask, A)]

  # 학습: 표준화 후 평균제곱오차를 전체 배치로 최소화
  def fit(self, X, y):
    c = self.cfg
    torch.set_num_threads(1)
    torch.manual_seed(c.nn_seed)
    n, d = c.gnn_nb + c.n_sup, 2 * c.n_items + 2
    x, mask = X[:, :n * d].reshape(-1, n, d), X[:, n * d:n * d + n].astype(bool)
    act = x[mask]
    self.mu, self.sd = act.mean(0), np.where(act.std(0) > 0, act.std(0), 1.0)
    self.ym, self.ys = y.mean(), y.std() or 1.0
    self.net = Net(d, c.gnn_hidden, c.gnn_layers)
    xs = self._unpack(X)
    t = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
    opt = torch.optim.Adam(self.net.parameters(), lr=c.nn_lr, weight_decay=self.wd)
    for _ in range(c.nn_epochs):
      opt.zero_grad()
      ((self.net(*xs) - t) ** 2).mean().backward()
      opt.step()
    return self

  # 가치 예측
  def predict(self, X):
    with torch.no_grad():
      return self.net(*self._unpack(X)).numpy().astype(float) * self.ys + self.ym


# weight decay 선택: nn_wds 중 공통 검증 분할의 평균제곱오차가 가장 작은 값으로 전체 데이터를 다시 맞춘다 (MLP와 같은 예산)
def select(cfg, X, y, g):
  return train.select(cfg, lambda wd: GNN(cfg, wd), cfg.nn_wds, X, y, g)
