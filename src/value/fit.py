"""Q̂ 근사 (N5): 선형(ridge 닫힌 해)과 MLP(torch)를 같은 입력, 같은 탐색 예산 8개로 학습한다.
목표는 고정 padding 대비 남은 누적 이윤 차이 A. 입력은 학습 자료 평균·표준편차로, 목표는 표준편차로 나눈다.
선택 기준은 검증 상태에서 Â 최대 후보의 실제 A 평균이다 (ref/decisions.md "Q̂ 탐색 예산").
"""
import numpy as np
import torch


class Q:
  """학습한 Q̂: 표준화 계수와 선형 계수 w 또는 MLP net을 담아 입력 행렬 → 후보별 Â를 낸다.
  Pool로 넘길 수 있게 함수 대신 객체로 둔다 (Windows spawn은 lambda를 넘기지 못함).
  w가 있으면 선형, 없으면 net을 쓴다.
  """

  def __init__(self, mu, sd, ys, w=None, net=None):
    self.mu, self.sd, self.ys, self.w, self.net = mu, sd, ys, w, net

  # 입력 행렬 → Â
  def __call__(self, X):
    Z = (X - self.mu) / self.sd
    if self.w is not None:
      return (np.c_[np.ones(len(Z)), Z] @ self.w) * self.ys
    with torch.no_grad():
      return self.net(torch.tensor(Z)).numpy()[:, 0] * self.ys


# 표준화 계수 (표준편차 0인 열은 1)
def scaler(X):
  sd = X.std(0)
  return X.mean(0), np.where(sd > 0, sd, 1.0)


# ridge (절편은 벌점 없음)
def ridge(X, Y, mu, sd, ys, lam):
  Z = np.c_[np.ones(len(X)), (X - mu) / sd]
  P = lam * np.eye(Z.shape[1])
  P[0, 0] = 0
  return Q(mu, sd, ys, w=np.linalg.solve(Z.T @ Z + P, Z.T @ (Y / ys)))


# MLP: 층 L개 × 숨은 H, ReLU, Adam(lr), 배치·에폭은 Cfg, 난수는 cfg.seed
def mlp(cfg, X, Y, mu, sd, ys, L, H, lr):
  torch.manual_seed(cfg.seed)
  mods, d = [], X.shape[1]
  for _ in range(L):
    mods += [torch.nn.Linear(d, H), torch.nn.ReLU()]
    d = H
  net = torch.nn.Sequential(*mods, torch.nn.Linear(d, 1)).double()
  opt = torch.optim.Adam(net.parameters(), lr=lr)
  Xt, Yt, g = torch.tensor((X - mu) / sd), torch.tensor(Y / ys)[:, None], torch.Generator().manual_seed(cfg.seed)
  for _ in range(cfg.mlp_epochs):
    for i in torch.randperm(len(Xt), generator=g).split(cfg.mlp_batch):
      opt.zero_grad()
      ((net(Xt[i]) - Yt[i]) ** 2).mean().backward()
      opt.step()
  return Q(mu, sd, ys, net=net.eval())


# 검증 점수: 상태마다 점수 최대 후보(동점은 앞)의 실제 A 평균, 그리고 예측 제곱오차
def score(v, Y):
  i = v.argmax(1)
  return dict(val=float(Y[np.arange(len(Y)), i].mean()), mse=float(((v - Y) ** 2).mean()))
