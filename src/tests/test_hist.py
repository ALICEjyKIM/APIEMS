"""거래 이력 환경(PROC15H, EnvH) 테스트.
이력 갱신식, 이탈·신규 처리(상속 없음), 이력 의존 재참여, 추가 난수 없음(h_w = 0이면 PROC15B와 같은 경로),
MILP의 참여자별 이력 항을 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg, PROC15B, PROC15H
from env.platform import Env, EnvH, Obs, worth
from env.response import HistRetention, Retention, response
from match.milp_solve import Policy
from bench.rule_split import rule

CFG = replace(Cfg(), **PROC15H)


# PROC15H는 EnvH와 HistRetention을 쓰고, h = h0이면 이력 없는 반응과 같으며 이력이 다르면 확률이 다르다
def test_dispatch_and_response():
  assert type(Env(CFG, 0)) is EnvH and isinstance(response(CFG), HistRetention)
  f, g = HistRetention(CFG), Retention(replace(Cfg(), **PROC15B))
  for e in (0.0, 1.0):
    assert f.prob("sup", 3.0, 50.0, e, CFG.h0) == pytest.approx(g.prob("sup", 3.0, 50.0, e))
    assert f.prob("sup", 3.0, 50.0, e, 0.9) > f.prob("sup", 3.0, 50.0, e, 0.1)
  assert f.prob("buy", 0.0, 1.0, 0.0, 1.0) - f.prob("buy", 0.0, 1.0, 0.0, 0.0) == pytest.approx(
    1 / (1 + np.exp(-(f.a + f.w / 2))) - 1 / (1 + np.exp(-(f.a - f.w / 2))))
  assert f.w == pytest.approx(CFG.h_w * f.e)
  assert f.prob("sup", f.rate("sup", 0.9, 0.2), 1.0, 1.0, 0.2) == pytest.approx(0.9)


# 갱신식: 남은 참여자 h_{t+1} = ρ h_t + (1 − ρ) × 경험, 품목 hI도 같은 식. 떠난 참여자 이력은 사라지고 신규는 h0
def test_update_and_lifecycle():
  env, pol = Env(CFG, 0), rule(CFG)
  env.reset()
  assert np.allclose(env.o.hb, CFG.h0) and np.allclose(env.o.hs, CFG.h0) and np.allclose(env.o.hi, 0)
  rho = CFG.h_rho
  for _ in range(12):
    o = env.o
    r = env.step(pol.act(o))
    hb, hs = dict(zip(o.bid, o.hb)), dict(zip(o.sid, o.hs))
    for i, h in zip(env.o.bid, env.o.hb):
      if i in hb:
        assert h == pytest.approx(rho * hb[i] + (1 - rho) * float(r["ok"][o.bid.index(i)]))
      else:
        assert h == CFG.h0
    for i, h in zip(env.o.sid, env.o.hs):
      if i in hs:
        assert h == pytest.approx(rho * hs[i] + (1 - rho) * float(r["alloc"][o.sid.index(i)]))
      else:
        assert h == CFG.h0
    assert np.allclose(env.o.hi, rho * o.hi + (1 - rho) * r["short"])
    gone = set(o.bid) - set(env.o.bid)
    assert not gone & set(env.hB) and not (set(o.sid) - set(env.o.sid)) & set(env.hS)
    assert set(env.hB) == set(env.o.bid) and set(env.hS) == set(env.o.sid)


# 자리를 새 후보가 채우면 새 ID이고 이력은 h0에서 시작한다 (떠난 공급자의 이력을 물려받지 않는다)
def test_no_inheritance():
  env, pol = Env(CFG, 0), Policy(CFG)
  env.reset()
  for _ in range(3):
    env.step(pol.act(env.o))
  old = env.o.sid[0]
  env.hS[old] = 0.99
  env._observe()
  seat = old % CFG.n_sup
  env.drop("sup", 0)
  assert old not in env.hS
  for _ in range(CFG.onboard):
    env.step(pol.act(env.o))
  new = env.sups[seat][0]
  assert new != old and new // CFG.n_sup == env.t
  assert env.o.hs[env.o.sid.index(new)] == CFG.h0


# 이력은 결정론적으로 갱신되고 난수를 더 쓰지 않는다: h_w = 0이면 PROC15B와 기간별 결과·참여자가 같다
def test_no_extra_randomness():
  a, b = Env(replace(CFG, h_w=0.0), 2), Env(replace(Cfg(), **PROC15B), 2)
  a.reset(), b.reset()
  pa, pb = rule(a.cfg), rule(b.cfg)
  for _ in range(CFG.T):
    ra, rb = a.step(pa.act(a.o)), b.step(pb.act(b.o))
    assert ra["profit"] == pytest.approx(rb["profit"]) and a.o.bid == b.o.bid and a.o.sid == b.o.sid
    assert np.array_equal(a.o.q, b.o.q) and np.array_equal(a.o.cap, b.o.cap)


# MILP 이력 항: 같은 공급자라도 이력이 높으면 잉여 0에서 재참여확률이 높아 배정 가치 c × gap_h가 달라진다
# 매 기간 최적해이고, 잉여가 0인 기간은 목적함수 = 이윤 + Σ c × 환경 재참여확률 (이력 반영)
@pytest.mark.parametrize("cv", [0.0, 30.0])
def test_hist_milp_rollout(cv):
  env = Env(CFG, 3)
  f = HistRetention(CFG)
  pol = Policy(CFG, split=(CFG.sh_buy, None), coef=lambda o: (np.zeros(len(o.bid)), np.full(len(o.sid), cv)))
  env.reset()
  for _ in range(CFG.T):
    o = env.o
    r = env.step(pol.act(o))
    assert r["profit"] >= -CFG.tol
    if r["ss"].sum() == 0:
      ws = worth(o)[1]
      assert pol.obj == pytest.approx(r["profit"] + cv * f.prob("sup", 0.0, ws, r["alloc"], o.hs).sum(), abs=1e-4)


# 손으로 만든 시장: 비싼 공급자(단위 마진 2 작음)에 1개 배정하면 c × gap_h를 얻는다. gap_h는 이력에 따라 다르므로 문턱 c가 달라진다
def test_hist_milp_threshold():
  f = HistRetention(CFG)
  lo, hi = f.gap("sup", 0.0), f.gap("sup", 1.0)
  c = 2.0 / ((lo + hi) / 2)
  out = {}
  for h in (0.0, 1.0):
    o = Obs(0, np.array([[10]]), np.array([[20.0]]), np.array([[10], [10]]), np.array([[10.0], [12.0]]), [0], [0, 1],
            hb=np.array([0.5]), hs=np.array([0.5, h]))
    x, u, v = Policy(CFG, coef=lambda o: (np.zeros(1), np.array([0.0, c]))).act(o)
    out[h] = x[0, 1].sum() > 0
  assert out[0.0] != out[1.0]
