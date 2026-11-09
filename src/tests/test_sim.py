"""기간 엔진 테스트: 재참여 확률표, 입찰가 불변, 공통 난수, 재참여 보상 장부, 잉여율, 이력·기준점 갱신, onboard, 교체 순번 키.
재참여 확률은 세 점과 기준점·제외 항의 정의를 손으로 계산한 값과 비교한다.
엔진 테스트는 기준 시장 반복 0을 짧게(T = 6) 돌린다.
"""
from copy import deepcopy
from dataclasses import replace, fields
import numpy as np
import pytest
from utils.params import Cfg
from utils.rng import ret_u
from env.ret import Ret, lg
from env.sim import Sim, run

CFG = replace(Cfg(), T=6)


def test_ret_table():
  c, R = CFG, Ret(CFG)
  p = lambda kind, e, x, h=CFG.h0, r=np.nan, u=0: float(R.prob(kind, e, x, h, r, u))
  for kind, rate in (("buy", c.r2_rate_buy), ("sup", c.r2_rate_sup)):
    assert p(kind, 0, 0) == pytest.approx(c.r2_p0)
    assert p(kind, 1, 0) == pytest.approx(c.r2_p1)
    assert p(kind, 1, rate) == pytest.approx(c.r2_p2)
    b1 = R.b1[kind]
    assert lg(p(kind, 1, 0.2, r=0.1)) - lg(p(kind, 1, 0.2)) == pytest.approx(c.eta * b1 * 0.1)
    assert lg(p(kind, 1, 0.1, r=0.12)) - lg(p(kind, 1, 0.1)) == pytest.approx(-c.eta * c.lam_loss * b1 * 0.02)
    assert p(kind, 1, 0.0, r=5.0) == pytest.approx(p(kind, 0, 0.0))
    assert p(kind, 0, 0.0, r=5.0) == pytest.approx(c.r2_p0)
  assert p("buy", 0, 0, u=1) == pytest.approx(c.ex_p)
  assert lg(p("buy", 0, 0, h=1.0)) - lg(p("buy", 0, 0)) == pytest.approx(c.h_w * R.b2 * (1 - c.h0))


# 1단계 관측은 이번 기간 입찰가를 읽지 않는다: 입찰가만 바꿔도 PreObs가 같다
def test_preobs_bid_invariant():
  s = Sim(CFG, 0)
  for _ in range(3):
    s.step(1.0, 0.25)
  a = s.pre()
  s.W, s.g = s.W * 3 + 1, s.g * 0.2
  b = s.pre()
  for f in fields(a):
    assert np.array_equal(getattr(a, f.name), getattr(b, f.name), equal_nan=True), f.name


# 공통 난수: 정책이 달라도 같은 기간의 신규 주문자와 같은 참여자의 재참여 난수가 같다
def test_common_random_numbers():
  A, B = run(CFG, 0, lambda o: (1.0, 0.0)), run(CFG, 0, lambda o: (1.5, 0.5))
  C = run(CFG, 0, lambda o: (1.0, 0.0))
  for a, c in zip(A, C):
    assert a["G"] == c["G"] and np.array_equal(a["zb"], c["zb"])
  ids = lambda R: [set(r["bid"]) for r in R]
  new = lambda R: [s - set().union(*ids(R)[:t]) if t else s for t, s in enumerate(ids(R))]
  assert new(A) == new(B)
  for r in A + B:
    for i, z, p in zip(r["bid"], r["zb"], r["pb"]):
      assert z == (ret_u(CFG, 0, "buy", i, r["t"]) < p)


# 재참여 보상: 총액 θ G를 ω × 이력(이번 거래 반영) 비중으로 나누고 상한 cbar ω 이내, 다음 기간 돌아온 사람에게만 지급
def test_reward_ledger():
  s, R = Sim(CFG, 0), []
  for t in range(CFG.T):
    o = s.pre()
    r = s.step(1.25, 0.5)
    w, k = np.r_[o.q @ s.mk.base, o.K @ s.mk.base], np.r_[r["kb"], r["ks"]]
    h = CFG.h_rho * np.r_[o.hb, o.hs] + (1 - CFG.h_rho) * np.r_[r["eb"], r["es"]]
    assert (k <= CFG.cbar * w + 1e-9).all() and r["cpn"] <= 0.5 * r["G"] + 1e-9
    free = (k < CFG.cbar * w - 1e-9) & (h > 0)
    assert np.allclose(k[free] / (w * h)[free], (k[free] / (w * h)[free]).mean())
    assert r["Pi"] == pytest.approx(r["G"] - r["paid"])
    if R:
      p = R[-1]
      owed = (p["kb"] * p["zb"]).sum() + (p["ks"] * p["zs"]).sum()
      assert o.owe == pytest.approx(owed) and r["paid"] == pytest.approx(owed)
    R.append(r)
  assert sum(r["cpn"] for r in R) > 0
  assert all(r["cpn"] == 0 for r in run(CFG, 0, lambda o: (1.25, 0.0)))


# 잉여율: 주문자는 묶음 가치, 공급자는 판 양의 원가로 나누고(못 팔면 0), 보상은 공개 가중치 ω로 나눠 더한다
def test_surplus_rate():
  s = Sim(CFG, 0)
  o, (W, g) = s.pre(), s.bids()
  r = s.step(1.0, 0.5)
  sold = (g * r["y"]).sum(1)
  assert ((sold > 0) == r["es"]).all() and (~r["es"]).any()
  xs = np.where(sold > 0, r["ss"] / np.where(sold > 0, sold, 1), 0) + CFG.cpn_w * r["ks"] / (o.K @ s.mk.base)
  assert np.allclose(r["xs"], xs) and np.allclose(r["xb"], r["sb"] / W + CFG.cpn_w * r["kb"] / (o.q @ s.mk.base))


# 이력·기준점: 남은 주문자의 h는 h_rho h + (1 − h_rho) e, 처음 거래한 주문자의 r은 그 기간 x
def test_history_reference():
  s = Sim(CFG, 0)
  r = s.step(1.0, 0.0)
  o = s.pre()
  pos = {i: k for k, i in enumerate(o.bid)}
  n = 0
  for i, e, x, z in zip(r["bid"], r["eb"], r["xb"], r["zb"]):
    if z:
      k = pos[i]
      assert o.hb[k] == pytest.approx(CFG.h_rho * CFG.h0 + (1 - CFG.h_rho) * e)
      assert (np.isnan(o.rb[k]) and not e) or o.rb[k] == pytest.approx(x)
      assert o.eb[k] == float(r["Ub"][list(r["bid"]).index(i)])
      n += e
  assert n > 0


# onboard: 기간 t에 떠난 자리는 t + 1에 비고 t + onboard에 새 번호 (t + onboard)·n_sup + j로 채워진다
def test_onboard():
  R, n = run(CFG, 0, lambda o: (1.0, 0.0)), 0
  for a, b in zip(R, R[1:]):
    for j, z in zip(a["slot"], a["zs"]):
      if not z:
        assert j not in b["slot"]
        t = a["t"] + CFG.onboard
        if t < CFG.T:
          k = list(R[t]["slot"]).index(j)
          assert R[t]["sid"][k] == t * CFG.n_sup + j
          n += 1
  assert n > 0


# 교체 순번 키(rep_key = slot): 자리가 언제 비든 그 자리의 첫 후보는 같은 번호 n_sup + j와 같은 프로필
def test_slot_key_replacement():
  cfg = replace(CFG, rep_key="slot")
  keep = lambda x, j=None: (np.ones(len(x.bid), bool), np.array([k != j for k in x.slot]))
  a, b = deepcopy(Sim(cfg, 0)), deepcopy(Sim(cfg, 0))
  a._next(*keep(a, 0))
  b._next(*keep(b))
  b._next(*keep(b, 0))
  for x in (a, b):
    while 0 not in x.S:
      x._next(*keep(x))
  assert a.S[0]["id"] == b.S[0]["id"] == cfg.n_sup
  assert np.array_equal(a.S[0]["prof"].qty, b.S[0]["prof"].qty) and np.allclose(a.S[0]["prof"].price, b.S[0]["prof"].price)
