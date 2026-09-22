"""MLP 가치 근사.
선형과 같은 시장 요약 지표를 입력으로, 은닉 nn_layers층 × nn_hidden(ReLU) 신경망으로 V(s)를 맞춘다.
weight decay는 nn_wds 중 공통 검증 분할의 평균제곱오차가 가장 작은 값을 고른다 (선형 λ와 같은 튜닝 예산).
"""
import numpy as np
import torch
from vfa import train


class MLP:
  """MLP 가치 근사.
  fit은 지표·목표를 학습 데이터로 표준화하고 Adam(weight decay wd)으로 전체 배치 nn_epochs회 학습한다 (seed nn_seed, 스레드 1).
  predict는 지표 행렬(행 = 상태)을 받아 가치 배열을 돌려준다.
  """

  def __init__(self, cfg, wd):
    self.cfg, self.wd = cfg, wd

  # 학습: 표준화(표준편차 0인 지표는 그대로) 후 평균제곱오차를 전체 배치로 최소화
  def fit(self, X, y):
    c = self.cfg
    torch.set_num_threads(1)
    torch.manual_seed(c.nn_seed)
    self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
    self.ym, self.ys = y.mean(), y.std() or 1.0
    dims = [X.shape[1]] + [c.nn_hidden] * c.nn_layers
    layers = [m for a, b in zip(dims[:-1], dims[1:]) for m in (torch.nn.Linear(a, b), torch.nn.ReLU())]
    self.net = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], 1))
    Z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
    t = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(self.net.parameters(), lr=c.nn_lr, weight_decay=self.wd)
    for _ in range(c.nn_epochs):
      opt.zero_grad()
      ((self.net(Z) - t) ** 2).mean().backward()
      opt.step()
    return self

  # 가치 예측
  def predict(self, X):
    with torch.no_grad():
      z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
      return self.net(z)[:, 0].numpy().astype(float) * self.ys + self.ym


# weight decay 선택: nn_wds 중 공통 검증 분할의 평균제곱오차가 가장 작은 값으로 전체 데이터를 다시 맞춘다 (vfa/train.select)
def select(cfg, X, y, g):
  return train.select(cfg, lambda wd: MLP(cfg, wd), cfg.nn_wds, X, y, g)
