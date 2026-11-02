"""Chu(2009) IPB padding 이중경매 청산 (Chu 5.2절, ref/design.md 3장).
padding 문제 V(B, J, Q)로 낙찰과 주문자 가격을 정하고, padding 없이 배정한 뒤 공급자 대금을 임계 가격으로 정한다.
Q = 0 청산을 한 번 더 풀어 padding 제외자(padding이 없었으면 거래했을 참여자)를 찾는다.
"""
import math
from dataclasses import dataclass
from functools import cache
import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class Out:
  """한 기간 청산 결과.
  win: 성립 주문, p: 주문자 가격(미성립 0), y: 배정량, crit: 공급자 임계 가격(거래하지 않으면 nan), R: 공급자 대금.
  G: 플랫폼 몫, sw: 실제 거래 이득, Ub·Uj: padding 제외 주문·공급자, n: MILP 풀이 횟수.
  """
  Q: np.ndarray
  win: np.ndarray
  p: np.ndarray
  y: np.ndarray
  crit: np.ndarray
  R: np.ndarray
  G: float
  sw: float
  Ub: np.ndarray
  Uj: np.ndarray
  n: int


# 품목별 padding 크기 Q_c = ⌈m × max(최대 공급자 물량, 한 주문 최대 요구량)⌉ (Chu Theorem 5, m ≥ 1)
def pad(cfg, m, q, K):
  assert m >= 1
  base = np.maximum(K.max(0), q.max(0))
  return np.array([math.ceil(m * b - cfg.tol) for b in base])


@cache
def env(cfg):
  e = gp.Env(empty=True)
  e.setParam("OutputFlag", 0)
  e.setParam("Seed", cfg.grb_seed)
  e.setParam("Threads", cfg.grb_threads)
  e.setParam("MIPGap", cfg.mip_gap)
  e.start()
  return e


class Prob:
  """padding 문제를 한 번 만들고 Q, 참여 주문, 원가, 공급량 범위만 바꿔 다시 푼다.
  z_c는 단가 a_max의 가상 공급자로 padding 문제가 항상 풀리게 한다 (Chu 6.1절 가정, 실제 배정에는 쓰이지 않음).
  목적식에 참여자 번호 순 미세 perturbation을 더해 동점을 깨고(Chu 각주 2), 값은 원래 계수로 계산한다.
  """

  def __init__(self, cfg, q, W, K, g, grp):
    nb, nc = q.shape
    assert W.max() < cfg.a_max
    self.cfg, self.q, self.W, self.K, self.g, self.n = cfg, q, W, K, g, 0
    self.ey = cfg.tol * (1 + np.arange(K.size)).reshape(K.shape) / K.size
    m = gp.Model(env=env(cfg))
    self.x = m.addMVar(nb, vtype=GRB.BINARY)
    self.y = m.addMVar(K.shape, vtype=GRB.INTEGER, ub=K)
    self.z = m.addMVar(nc, vtype=GRB.INTEGER, ub=0)
    self.bal = m.addConstr(self.y.sum(0) + self.z - q.T @ self.x == 0)
    for k in np.unique(grp):
      if (grp == k).sum() > 1:
        m.addConstr(self.x[np.flatnonzero(grp == k)].sum() <= 1)
    ex = cfg.tol * 0.5 ** np.arange(nb)
    m.setObjective((W - cfg.f_ship + ex) @ self.x - (self.y * (g + self.ey)).sum() - (cfg.a_max + cfg.tol) * self.z.sum(), GRB.MAXIMIZE)
    self.m = m

  # 패딩 Q, 참여 주문 on, 원가 g, 공급량 범위 [lo, hi]에서 풀어 (원래 계수의 값, x, y)
  def solve(self, Q, on, g=None, lo=None, hi=None):
    g = self.g if g is None else g
    self.x.UB = on.astype(float)
    self.bal.RHS = Q
    self.z.UB = Q
    self.y.Obj = -(g + self.ey)
    self.y.LB = np.zeros(self.K.shape) if lo is None else lo
    self.y.UB = self.K if hi is None else hi
    self.m.optimize()
    assert self.m.Status == GRB.OPTIMAL
    self.n += 1
    x, y, z = np.round(self.x.X).astype(bool), np.round(self.y.X).astype(int), np.round(self.z.X)
    return (self.W - self.cfg.f_ship) @ x - (g * y).sum() - self.cfg.a_max * z.sum(), x, y

  # 공급자 j의 품목 c 임계 가격: padding 문제에서 y_jc = K_jc인 가장 높은 원가 신고. 꽉 채운 값 A − gK와 덜 채운 값 B(g)의 교점을 위에서 찾는다
  def crit(self, Q, j, c):
    on, k = np.ones(len(self.W), bool), self.K[j, c]
    g0, lo, hi = self.g.copy(), np.zeros_like(self.K), self.K.copy()
    g0[j, c], lo[j, c] = 0, k
    A = self.solve(Q, on, g0, lo)[0]
    hi[j, c], g = k - 1, self.cfg.a_max
    for _ in range(100):
      g0[j, c] = g
      b, _, y = self.solve(Q, on, g0, None, hi)
      if A - g * k >= b - self.cfg.tol:
        return g
      g = (A - b - g * y[j, c]) / (k - y[j, c])
    raise AssertionError("임계 가격 반복이 끝나지 않음")


# 입찰 (q, W, K, g)와 padding 배수 m으로 청산한다. grp는 같은 주문자의 묶음(최대 하나 성립, Chu 5.3 예제용)
def clear(cfg, q, W, K, g, m, grp=None):
  nb, nc = q.shape
  grp = np.arange(nb) if grp is None else grp
  P, Q, on = Prob(cfg, q, W, K, g, grp), pad(cfg, m, q, K), np.ones(nb, bool)
  V, win, _ = P.solve(Q, on)
  p = np.zeros(nb)
  for b in np.flatnonzero(win):
    p[b] = W[b] - (V - P.solve(Q, grp != grp[b])[0])
  Z = np.zeros(nc)
  v, x, y = P.solve(Z, win)
  assert (x == win).all()
  crit, R = np.full(K.shape, np.nan), np.zeros(K.shape)
  for j, c in zip(*np.nonzero(y)):
    crit[j, c] = P.crit(Q, j, c)
    assert crit[j, c] >= g[j, c] - cfg.tol
    gc = g.copy()
    gc[j, c] = crit[j, c]
    R[j, c] = g[j, c] * y[j, c] + v - P.solve(Z, win, gc)[0]
  G = (p[win] - cfg.f_ship).sum() - R.sum()
  assert G >= -cfg.tol * (nb + K.size)
  _, x0, y0 = P.solve(Z, on)
  Ub, Uj = x0 & ~win, (y0.sum(1) > 0) & (y.sum(1) == 0)
  return Out(Q, win, p, y, crit, R, G, v, Ub, Uj, P.n)
