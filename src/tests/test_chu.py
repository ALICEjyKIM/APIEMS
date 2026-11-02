"""Chu IPB 청산 테스트: 논문·보고서 예제, 전수 열거 기준과의 일치, 기준 시장에서의 성질.
전수 열거는 주문 부분집합마다 품목별로 싼 단위부터 채우는 독립 구현이다 (임계 가격은 이분탐색).
기준 시장에서는 m 후보마다 G ≥ 0, 개인 합리성, 배정의 수요·용량 일치를 확인한다.
"""
import itertools
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from utils import market
from mech.chu import clear, pad

CFG = replace(Cfg(), f_ship=0.0)


# 전수 열거: 주문 부분집합마다 품목별 싼 단위부터 채워 (값, 성립 집합, 공급량)
def brute(cfg, q, W, K, g, Q, on):
  nb, best = len(W), (-np.inf, None, None)
  for r in range(nb + 1):
    for sub in itertools.combinations(np.flatnonzero(on), r):
      x = np.zeros(nb, bool)
      x[list(sub)] = True
      need, y, v = q[x].sum(0) + Q, np.zeros_like(K), (W[x] - cfg.f_ship).sum()
      for c in range(q.shape[1]):
        n = need[c]
        for j in np.argsort(g[:, c], kind="stable"):
          y[j, c] = min(K[j, c], n)
          n -= y[j, c]
        v -= (g[:, c] * y[:, c]).sum() + cfg.a_max * n
      if v > best[0] + 1e-9:
        best = (v, x, y)
  return best


# 전수 열거로 Chu IPB 청산 (임계 가격은 이분탐색)
def brute_clear(cfg, q, W, K, g, m):
  nb, on, Q = len(W), np.ones(len(W), bool), pad(cfg, m, q, K)
  V, win, _ = brute(cfg, q, W, K, g, Q, on)
  p = np.zeros(nb)
  for b in np.flatnonzero(win):
    p[b] = W[b] - (V - brute(cfg, q, W, K, g, Q, np.arange(nb) != b)[0])
  Z = np.zeros(q.shape[1], int)
  v, _, y = brute(cfg, q, W, K, g, Z, win)
  R = np.zeros(K.shape)
  for j, c in zip(*np.nonzero(y)):
    lo, hi, gc = g[j, c], cfg.a_max, g.copy()
    for _ in range(60):
      gc[j, c] = (lo + hi) / 2
      full = brute(cfg, q, W, K, gc, Q, on)[2][j, c] == K[j, c]
      lo, hi = (gc[j, c], hi) if full else (lo, gc[j, c])
    gc[j, c] = lo
    R[j, c] = g[j, c] * y[j, c] + v - brute(cfg, q, W, K, gc, Z, win)[0]
  return win, p, y, R


def test_pad():
  q, K = np.array([[2, 0], [0, 1]]), np.array([[10, 0], [1, 1]])
  assert pad(CFG, 1.0, q, K).tolist() == [10, 1]
  assert pad(CFG, 1.1, q, K).tolist() == [11, 2]
  assert pad(CFG, 1.5, q, K).tolist() == [15, 2]


# Chu 5.3: 주문자 1은 A 2개(9) 또는 B 1개(3). 공급자 1 A 2개(1), 2·3 A 1개(2, 4), 4·5 B 1개(1)
def test_chu_53():
  q, W = np.array([[2, 0], [0, 1]]), np.array([9.0, 3.0])
  K = np.array([[2, 0], [1, 0], [1, 0], [0, 1], [0, 1]])
  g = np.array([[1.0, 0], [2, 0], [4, 0], [0, 1], [0, 1]])
  o = clear(CFG, q, W, K, g, 1.0, grp=np.array([0, 0]))
  assert o.Q.tolist() == [2, 1]
  assert o.win.tolist() == [True, False]
  assert o.p[0] == pytest.approx(6)
  assert o.y[:, 0].tolist() == [2, 0, 0, 0, 0]
  assert o.crit[0, 0] == pytest.approx(3)
  assert o.R[0, 0] == pytest.approx(5)
  assert o.G == pytest.approx(1)


# 보고서 6.1: Q = (2, 2)에서 주문자 1만 20에 성립, 공급자 1(A)이 10, 공급자 3(B)이 4를 받아 플랫폼 몫 6
def test_report_61():
  q, W = np.array([[2, 1], [1, 0], [0, 2], [1, 1]]), np.array([30.0, 9, 14, 11])
  K = np.array([[2, 1], [2, 0], [0, 2]])
  g = np.array([[3.0, 4], [5, 0], [0, 2]])
  o = clear(CFG, q, W, K, g, 1.0)
  assert o.Q.tolist() == [2, 2]
  assert o.win.tolist() == [True, False, False, False]
  assert o.p[0] == pytest.approx(20)
  assert o.y.tolist() == [[2, 0], [0, 0], [0, 1]]
  assert o.crit[0, 0] == pytest.approx(13) and o.crit[2, 1] == pytest.approx(16)
  assert o.R[0, 0] == pytest.approx(10) and o.R[2, 1] == pytest.approx(4)
  assert o.G == pytest.approx(6) and o.sw == pytest.approx(22)
  assert o.Ub.tolist() == [False, True, True, False]
  assert o.Uj.tolist() == [False, True, False]


# 무작위 작은 시장에서 MILP 청산과 전수 열거 청산이 같은 낙찰·가격·배정·대금을 낸다
@pytest.mark.parametrize("s", range(8))
def test_vs_brute(s):
  g_ = np.random.default_rng(s)
  nb, ns, nc = 5, 4, 2
  q = g_.integers(0, 4, (nb, nc))
  q[q.sum(1) == 0, 0] = 1
  K = g_.integers(0, 4, (ns, nc))
  g = np.round(g_.uniform(1, 5, (ns, nc)), 3) * (K > 0)
  W = np.round(q.sum(1) * g_.uniform(3, 8, nb), 3)
  for m in CFG.m_grid:
    o = clear(CFG, q, W, K, g, m)
    win, p, y, R = brute_clear(CFG, q, W, K, g, m)
    assert (o.win == win).all() and (o.y == y).all()
    assert o.p == pytest.approx(p, abs=1e-6)
    assert o.R == pytest.approx(R, abs=1e-6)


# 기준 시장(신규 주문자 두 기간 분량, 기간 0 공급자)에서 m 후보마다 G ≥ 0, 개인 합리성, 수요·용량 일치
def test_market_props():
  cfg = Cfg()
  mk = market.make(cfg, 0)
  bs = market.new_buyers(cfg, mk, 0, 0) + market.new_buyers(cfg, mk, 0, 1)
  q, W = np.array([b.qty for b in bs]), np.array([(b.qty * b.price).sum() for b in bs])
  K, g = np.array([s.qty for s in mk.sups]), np.array([s.price for s in mk.sups])
  for m in cfg.m_grid:
    o = clear(cfg, q, W, K, g, m)
    assert o.G >= -1e-6
    assert (o.p[o.win] <= W[o.win] + 1e-6).all()
    assert (o.R >= g * o.y - 1e-6).all()
    assert (o.y.sum(0) == q[o.win].sum(0)).all() and (o.y <= K).all()
