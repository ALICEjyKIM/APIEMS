"""Q̂ 근사 (N5): 선형(ridge 닫힌 해)과 MLP(torch)를 같은 입력, 같은 탐색 예산 8개로 학습한다.
1차 목표는 고정 padding 대비 남은 누적 이윤 차이 A, 2차(TD)는 기간마다 Π + 행동 정책의 다음 상태 가치 Q̂_{t+1}이다. 입력은 평균·표준편차로, 목표는 표준편차로 나눈다.
선택 기준은 검증 상태에서 Â 최대 후보의 실제 A 평균이다 (ref/decisions.md "Q̂ 탐색 예산", "TD 학습 (2차)").
"""
from copy import deepcopy
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
    Z = (np.asarray(X, float) - self.mu) / self.sd
    if self.w is not None:
      return (np.c_[np.ones(len(Z)), Z] @ self.w) * self.ys
    with torch.no_grad():
      return self.net(torch.tensor(Z)).numpy()[:, 0] * self.ys


class QT:
  """기간별 Q̂ 묶음 (TD): 입력 첫 열 t/T로 그 기간의 Q̂_t를 골라 쓰고, 그 기간 학습 목표의 범위 [lo, hi]로 자른다.
  자르지 않으면 학습에 드문 다음 상태에서 외삽한 값이 max로 뽑혀 앞 기간으로 커진다.
  한 번 호출에 들어오는 행렬은 한 상태의 후보들이라 기간이 같다. Q처럼 Pool로 넘길 수 있다.
  """

  def __init__(self, qs, T, lo, hi):
    self.qs, self.T, self.lo, self.hi = qs, T, lo, hi

  # 입력 행렬 → Â (그 기간의 Q̂_t, 목표 범위로 자름). GNN 입력 (행렬, 그래프 목록)도 받는다
  def __call__(self, X):
    s = int(round((X[0] if isinstance(X, tuple) else X)[0, 0] * self.T))
    return np.clip(self.qs[s](X), self.lo[s], self.hi[s])


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


# MLP: 층 L개 × 숨은 H, ReLU, Adam(lr), 배치·에폭은 Cfg, 난수는 cfg.seed. hold = (X, Y)를 주면 10에폭마다 그 제곱오차가 가장 작은 시점을 쓴다
def mlp(cfg, X, Y, mu, sd, ys, L, H, lr, hold=None):
  torch.manual_seed(cfg.seed)
  mods, d = [], X.shape[1]
  for _ in range(L):
    mods += [torch.nn.Linear(d, H), torch.nn.ReLU()]
    d = H
  net = torch.nn.Sequential(*mods, torch.nn.Linear(d, 1)).double()
  opt = torch.optim.Adam(net.parameters(), lr=lr)
  Xt, Yt, g = torch.tensor((X - mu) / sd), torch.tensor(Y / ys)[:, None], torch.Generator().manual_seed(cfg.seed)
  best = (np.inf, None)
  for e in range(cfg.mlp_epochs):
    for i in torch.randperm(len(Xt), generator=g).split(cfg.mlp_batch):
      opt.zero_grad()
      ((net(Xt[i]) - Yt[i]) ** 2).mean().backward()
      opt.step()
    if hold is not None and (e + 1) % 10 == 0:
      err = float(((Q(mu, sd, ys, net=net)(hold[0]) - hold[1]) ** 2).mean())
      best = min(best, (err, deepcopy(net.state_dict())), key=lambda b: b[0])
  if best[1] is not None:
    net.load_state_dict(best[1])
  return Q(mu, sd, ys, net=net.eval())


# TD 목표: Π + 행동 정책의 다음 상태 가치 (1 − eps) q(S', 고정 padding) + eps 평균_a' q(S', a') (q가 없으면 마지막 기간이라 Π)
def target(q, P, Xn, eps):
  if q is None:
    return P
  n, k = P.shape
  v = q(Xn.reshape(-1, Xn.shape[-1])).reshape(n, k, -1)
  return P + (1 - eps) * v[:, :, 0] + eps * v.mean(2)


# 검증 점수: 상태마다 점수 최대 후보(동점은 앞)의 실제 A 평균, 그리고 예측 제곱오차
def score(v, Y):
  i = v.argmax(1)
  return dict(val=float(Y[np.arange(len(Y)), i].mean()), mse=float(((v - Y) ** 2).mean()))
