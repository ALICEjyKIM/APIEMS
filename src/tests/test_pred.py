"""시나리오 입찰가 테스트: 경험분포(uh, ch)가 과거 입찰과 같고, 시나리오가 그 분포에서만 나오며, 같은 관측이면 같은 시나리오이고, 시나리오 정책이 입찰가를 읽지 않는지.
참고용 시나리오는 재참여자 자리에 지난 신고가를 그대로 둔다.
엔진은 기준 시장 반복 0을 짧게(T = 6) 돌린다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.sim import Sim
from mech.chu import clear
from policy.base import scen, scen_ind, saa, table, gains
from env.sim import cands

CFG = replace(Cfg(), T=6, saa_k=4)


# 경험분포: uh는 과거 주문자 비율 전부, ch[c]는 품목 c 자격 공급자의 과거 원가 전부
def test_hist():
  s = Sim(CFG, 0)
  q, W, K, g, base = s.q.copy(), s.W.copy(), s.K.copy(), s.g.copy(), s.mk.base
  s.step(1.0, 0.0)
  o = s.pre()
  assert o.uh == pytest.approx(W / (q @ base))
  for c in range(CFG.n_items):
    assert o.ch[o.ci == c] == pytest.approx(g[K[:, c] > 0, c])


# 시나리오: 주문자 값은 uh의 비율 × 기준가 묶음, 공급자 값은 자격 품목에만 그 품목(표본이 적으면 전체 비율) 분포에서, 같은 관측이면 같은 시나리오
def test_scen():
  s = Sim(CFG, 0)
  s.step(1.0, 0.0)
  s.step(1.0, 0.0)
  o = s.pre()
  S = scen(CFG, o, 3)
  assert len(S) == 3 and len({tuple(W) for W, _ in S}) == 3
  pool = o.ch / o.base[o.ci]
  for W, g in S:
    u = W / (o.q @ o.base)
    assert all(np.isclose(o.uh, x).any() for x in u)
    assert (g[o.K == 0] == 0).all()
    for c in range(CFG.n_items):
      src = o.ch[o.ci == c] if (o.ci == c).sum() >= CFG.saa_min else pool * o.base[c]
      assert all(np.isclose(src, x).any() for x in g[o.K[:, c] > 0, c])
  assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(S, scen(CFG, o, 3)))
  for W, g in scen_ind(CFG, o, 2):
    k = ~np.isnan(o.W0)
    assert np.array_equal(W[k], o.W0[k]) and np.array_equal(g[(o.K > 0) & ~np.isnan(o.g0)], o.g0[(o.K > 0) & ~np.isnan(o.g0)])


# 시나리오 정책: 기록 없는 기간 0은 고정, 이후는 시나리오 평균 플랫폼 몫 최대 m, 이번 기간 입찰가를 바꿔도 같은 선택
def test_saa():
  s, pol = Sim(CFG, 0), saa(CFG)
  assert pol(s.pre()) == (1.0, 0.0)
  for _ in range(3):
    s.step(*pol(s.pre()))
  o = s.pre()
  G = [np.mean([clear(CFG, o.q, W, o.K, g, m).G for W, g in scen(CFG, o, CFG.saa_k)]) for m in CFG.m_grid]
  m, th = pol(o)
  assert th == 0.0 and m == CFG.m_grid[int(np.argmax(G))]
  s.W, s.g = s.W * 3 + 1, s.g * 0.2
  assert pol(s.pre()) == (m, th)


# 장기 표 규칙: 기간 0은 고정, 이후 손실 표가 0이면 시나리오 한 기간 최적과 같은 선택, 큰 m에 큰 손실을 주면 작은 m을 고르고, 입찰가를 바꿔도 같은 선택
def test_table():
  s = Sim(CFG, 0)
  ms = sorted({m for m, _ in cands(CFG)})
  z = [[0.0] * len(ms)]
  assert table(CFG, z, [0])(s.pre()) == (1.0, 0.0)
  for _ in range(3):
    s.step(1.0, 0.0)
  o = s.pre()
  assert table(CFG, z, [0])(o) == saa(CFG)(o)
  big = [[0.0] + [-1e9] * (len(ms) - 1)]
  assert table(CFG, big, [0])(o) == (1.0, 0.0)
  G = gains(CFG, o, ms, "mean")
  assert table(CFG, z, [0], "mean")(o) == (ms[int(np.argmax(G - G[0]))], 0.0)
  m = table(CFG, z, [0])(o)
  s.W, s.g = s.W * 3 + 1, s.g * 0.2
  assert table(CFG, z, [0])(s.pre()) == m
