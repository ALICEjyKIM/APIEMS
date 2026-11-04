"""비교 정책 테스트: 시장 통계, 추정 입찰가, 한 기간 최적의 선택 규칙과 입찰가 불변, 공통 난수 rollout, Q̂ 입력과 Q̂ 정책.
한 기간 최적은 추정 입찰가로 후보 m마다 청산한 플랫폼 몫을 직접 비교한 값과 같아야 한다.
엔진은 기준 시장 반복 0을 짧게(T = 6) 돌린다.
"""
import pickle
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.sim import Sim, run, cands
from mech.chu import clear
from policy.base import fixed, est, myopic
from policy.qpol import qpol
from value.feat import feats, plan

CFG = replace(Cfg(), T=6)


# vbar: 묶음 가치를 품목 기준가 비중으로 나눈 단위 가치의 품목별 평균, cbar: 자격 공급자 원가의 품목별 평균
def test_stats():
  s = Sim(CFG, 0)
  q, W, K, g, base = s.q.copy(), s.W.copy(), s.K.copy(), s.g.copy(), s.mk.base
  s.step(1.0, 0.0)
  o = s.pre()
  u = W / (q @ base)
  for c in range(CFG.n_items):
    b, j = q[:, c] > 0, K[:, c] > 0
    assert np.isnan(o.vbar[c]) if not b.any() else o.vbar[c] == pytest.approx(u[b].mean() * base[c])
    assert np.isnan(o.cbar[c]) if not j.any() else o.cbar[c] == pytest.approx(g[j, c].mean())


# 추정 입찰가: 주문자는 Σ q vbar, 공급자는 자격 품목에만 cbar, 기록 없는 품목은 기준가 × 평균 비율
def test_est():
  s = Sim(CFG, 0)
  s.step(1.0, 0.0)
  o = s.pre()
  o.vbar[0], o.cbar[0] = np.nan, np.nan
  W, g = est(o)
  rv, rc = np.nanmean(o.vbar / o.base), np.nanmean(o.cbar / o.base)
  v, c = np.where(np.isnan(o.vbar), rv * o.base, o.vbar), np.where(np.isnan(o.cbar), rc * o.base, o.cbar)
  assert v[0] == pytest.approx(rv * o.base[0]) and W == pytest.approx(o.q @ v)
  assert (g[o.K == 0] == 0).all() and g[o.K > 0] == pytest.approx(np.broadcast_to(c, o.K.shape)[o.K > 0])


# 한 기간 최적: 기록이 없는 기간 0은 고정 padding, 이후는 추정 입찰가 청산의 플랫폼 몫 최대 m, θ = 0
def test_myopic():
  s, pol = Sim(CFG, 0), myopic(CFG)
  assert pol(s.pre()) == (1.0, 0.0)
  for _ in range(4):
    s.step(*pol(s.pre()))
    o = s.pre()
    W, g = est(o)
    G = [clear(CFG, o.q, W, o.K, g, m).G for m in CFG.m_grid]
    m, th = pol(o)
    assert th == 0.0 and m == CFG.m_grid[int(np.argmax(G))]
  # 1단계는 입찰가를 읽지 않는다: 이번 기간 입찰가만 바꿔도 선택이 같다
  s.W, s.g = s.W * 3 + 1, s.g * 0.2
  assert pol(s.pre()) == (m, th)


# 공통 난수 rollout: 정책이 달라도 기간마다 같은 신규 주문자가 오고, 같은 정책은 같은 경로를 낸다
def test_rollout_crn():
  A, B, C = run(CFG, 0, fixed(1.0, 0.0)), run(CFG, 0, myopic(CFG)), run(CFG, 0, fixed(1.0, 0.0))
  ids = lambda R: [set(r["bid"]) for r in R]
  new = lambda R: [s - set().union(*ids(R)[:t]) if t else s for t, s in enumerate(ids(R))]
  assert new(A) == new(B)
  assert [r["Pi"] for r in A] == [r["Pi"] for r in C]
  assert all(r["m"] == 1.0 and r["th"] == 0.0 for r in A)


# Q̂ 입력: 후보 12개 × (상태 17 + 후보 11 + m × 상태 + θ × 상태), m·θ 열이 후보와 같고 입찰가를 바꿔도 같다
def test_feats():
  s = Sim(CFG, 0)
  s.step(1.0, 0.0)
  X = feats(CFG, s.pre())
  assert X.shape == (len(cands(CFG)), 17 * 3 + 11)
  assert [tuple(x[17:19]) for x in X] == cands(CFG)
  s.W, s.g = s.W * 3 + 1, s.g * 0.2
  assert np.array_equal(X, feats(CFG, s.pre()))


# 예상 청산(plan)을 따로 저장해 pickle 왕복해도 같은 Q̂ 입력이 나온다 (원재료 저장, decisions.md "질문 3 비교 범위")
def test_plan_raw():
  s = Sim(CFG, 0)
  s.step(1.0, 0.0)
  o = s.pre()
  o2, P = pickle.loads(pickle.dumps((o, plan(CFG, o))))
  assert np.array_equal(feats(CFG, o), feats(CFG, o2, P))
  assert all(p.p is None and p.R is None for p in P.values())


# Q̂ 정책: 점수 최대 후보, 동점은 앞 후보, only로 제한, 기간 0은 고정 padding
def test_qpol():
  s = Sim(CFG, 0)
  assert qpol(CFG, lambda X: X[:, 17])(s.pre()) == (1.0, 0.0)
  s.step(1.0, 0.0)
  o = s.pre()
  assert qpol(CFG, lambda X: X[:, 17] + X[:, 18])(o) == (1.5, 0.5)
  assert qpol(CFG, lambda X: X[:, 17] + X[:, 18], only=lambda m, th: th == 0)(o) == (1.5, 0.0)
  assert qpol(CFG, lambda X: X[:, 17] + X[:, 18], only=lambda m, th: m == 1)(o) == (1.0, 0.5)
  assert qpol(CFG, lambda X: np.zeros(len(X)))(o) == (1.0, 0.0)
