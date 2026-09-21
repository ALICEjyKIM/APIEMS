"""환경 정산, 기간 전이, rollout 테스트 (슬라이스 1).
손으로 계산 가능한 작은 시장에서 주문 성립 조건, 분할 공급, 용량 검사, 이윤 구성요소를 확인한다.
기간 전이는 공통 난수와 참여 규칙(주문자 1기간, 공급자 유지)을, rollout은 성과 지표 집계를 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Obs, Env, settle
from env.response import Logistic
from match.milp_solve import Policy
from match.interface import rollout

CFG = Cfg()
SH = 1 - CFG.sh_buy - CFG.sh_sup


# 주문자 1명(품목 0을 10개, 제안가격 20), 공급자 2명(각자 품목 0을 6개, 가격 10·12)
def split_obs():
  q, p = np.array([[10, 0]]), np.array([[20.0, 0]])
  cap, c = np.array([[6, 0], [6, 0]]), np.array([[10.0, 0], [12.0, 0]])
  return Obs(0, q, p, cap, c, [0], [0, 1])


# 주문자 1명(품목 0·1을 5개씩, 제안가격 20·30), 공급자 1명(품목 0·1을 5개씩, 가격 10·10)
def bundle_obs():
  q, p = np.array([[5, 5]]), np.array([[20.0, 30.0]])
  cap, c = np.array([[5, 5]]), np.array([[10.0, 10.0]])
  return Obs(0, q, p, cap, c, [0], [0])


# 분할 공급 결정(6개·4개)과 고정 비율 잉여: 주문 마진 92, 공급자별 마진 60·32
X_SPLIT = np.array([[[6, 0], [4, 0]]])
SB_RULE, SS_RULE = np.array([CFG.sh_buy * 92]), np.array([CFG.sh_sup * 60, CFG.sh_sup * 32])


# 한 품목을 두 공급자가 6개·4개로 나눠 채우면 주문이 성립하고 이윤을 손으로 계산한 값과 같다
def test_split_supply_fulfills():
  r = settle(CFG, split_obs(), (X_SPLIT, SB_RULE, SS_RULE))
  gross = 10 * 6 + 8 * 4
  assert r["ok"].tolist() == [True] and r["n_ok"] == 1
  assert r["rev"] == pytest.approx(SH * gross)
  assert r["ship"] == CFG.f_ship
  assert r["profit"] == pytest.approx(SH * gross - CFG.f_ship)
  assert r["opp"] == 0
  assert r["sb"] == pytest.approx([CFG.sh_buy * gross])
  assert r["ss"] == pytest.approx([CFG.sh_sup * 60, CFG.sh_sup * 32])


# 배분 잉여(주문자 + 공급자 + 플랫폼 이윤)의 합 = 거래잉여(마진 − 배송비), 잉여 배분이 달라도 같다
@pytest.mark.parametrize("sb,ss", [(SB_RULE, SS_RULE), (np.zeros(1), np.zeros(2)), (np.array([50.0]), np.array([10.0, 7.0]))])
def test_surplus_conserved(sb, ss):
  r = settle(CFG, split_obs(), (X_SPLIT, sb, ss))
  assert r["sb"].sum() + r["ss"].sum() + r["profit"] == pytest.approx(92 - CFG.f_ship)
  assert r["profit"] == pytest.approx(92 - CFG.f_ship - sb.sum() - ss.sum())


# 요구수량을 다 채우지 못하면 불성립: 수익·배송비 0, 기회손실 = 최저가 공급자 기준 거래잉여(마진 − 배송비)
def test_partial_not_fulfilled():
  r = settle(CFG, split_obs(), (np.array([[[6, 0], [0, 0]]]), np.zeros(1), np.zeros(2)))
  assert r["ok"].tolist() == [False]
  assert r["rev"] == 0 and r["ship"] == 0 and r["profit"] == 0
  assert r["sb"].sum() == 0 and r["ss"].sum() == 0
  assert r["opp"] == pytest.approx(max(0, 10 * (20 - 10) - CFG.f_ship))


# 묶음 주문은 모든 품목이 확보될 때만 성립한다
def test_bundle_all_items():
  o = bundle_obs()
  assert settle(CFG, o, (np.array([[[5, 0]]]), np.zeros(1), np.zeros(1)))["ok"].tolist() == [False]
  r = settle(CFG, o, (np.array([[[5, 5]]]), np.array([CFG.sh_buy * 150]), np.array([CFG.sh_sup * 150])))
  assert r["ok"].tolist() == [True]
  assert r["profit"] == pytest.approx(SH * (10 * 5 + 20 * 5) - CFG.f_ship)


# 공급자 용량 초과, 음수 수량, 정수가 아닌 수량은 거부한다
@pytest.mark.parametrize("x", [
  np.array([[[7, 0], [3, 0]]]),
  np.array([[[11, 0], [-1, 0]]]),
  np.array([[[6.0, 0], [4.0, 0]]]),
  np.array([[[0, 1], [0, 0]]]),
])
def test_invalid_decision_rejected(x):
  with pytest.raises(AssertionError):
    settle(CFG, split_obs(), (x, np.zeros(1), np.zeros(2)))


# 음수 잉여, 마진을 넘는 잉여, 불성립 주문자에게 준 잉여, 플랫폼 적자(보조금)는 거부한다
@pytest.mark.parametrize("x,sb,ss", [
  (X_SPLIT, np.array([-1.0]), np.zeros(2)),
  (X_SPLIT, np.array([93.0]), np.zeros(2)),
  (X_SPLIT, np.zeros(1), np.array([61.0, 0.0])),
  (np.array([[[6, 0], [0, 0]]]), np.array([1.0]), np.zeros(2)),
  (X_SPLIT, np.array([60.0]), np.array([10.0, 0.0])),
])
def test_invalid_surplus_rejected(x, sb, ss):
  with pytest.raises(AssertionError):
    settle(CFG, split_obs(), (x, sb, ss))


# 주문자가 없는 기간도 정산된다
def test_no_buyers():
  o = Obs(0, np.zeros((0, 2), int), np.zeros((0, 2)), np.array([[6, 0]]), np.array([[10.0, 0]]), [], [0])
  r = settle(CFG, o, (np.zeros((0, 1, 2), int), np.zeros(0), np.zeros(1)))
  assert r["profit"] == 0 and r["n_ok"] == 0 and r["opp"] == 0


# 관측은 인스턴스의 공급자와 신규 주문자로 구성된다
def test_reset_obs():
  env = Env(CFG, 0)
  o = env.reset()
  assert o.t == 0
  assert o.q.shape == o.p.shape == (len(o.bid), CFG.n_items)
  assert o.cap.shape == o.c.shape == (CFG.n_sup, CFG.n_items)
  assert np.array_equal(o.cap, env.inst.sup_cap)
  assert (o.q.sum(1) > 0).all()


# 공통 난수: 다음 기간 시장은 이번 기간 결정과 무관하다. 주문자는 떠나고 공급자는 모두 남는다
def test_transition_common_random():
  a, b = Env(CFG, 0), Env(CFG, 0)
  a.reset(), b.reset()
  for _ in range(5):
    oa, B, J = a.o, len(a.o.bid), len(a.o.sid)
    ra, rb = a.step((np.zeros((B, J, CFG.n_items), int), np.zeros(B), np.zeros(J))), b.step(Policy(CFG).act(b.o))
    assert rb["n_ok"] > 0
    assert ra["stay_buy"] == 0 and ra["stay_sup"] == ra["n_sup"] == CFG.n_sup
    assert set(a.o.bid).isdisjoint(oa.bid)
    for k in ("q", "p", "cap", "c"):
      assert np.array_equal(getattr(a.o, k), getattr(b.o, k))
    assert a.o.bid == b.o.bid and a.o.sid == b.o.sid


# rollout: 누적 이윤 = 수익 − 배송비, 주문 충족률 = 성립 주문 / 전체 주문, 유지율(주문자 0, 공급자 1: 재참여 반응 전)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rollout_metrics(k):
  cfg = replace(CFG, items_per_order=k)
  env, pol = Env(cfg, 0), Policy(cfg)
  env.reset()
  pol.reset(0)
  rs = [env.step(pol.act(env.o)) for _ in range(cfg.T)]
  out = rollout(cfg, Policy(cfg), 0)
  assert out["profit"] == pytest.approx(sum(r["profit"] for r in rs))
  assert out["profit"] == pytest.approx(out["rev"] - out["ship"])
  assert out["opp"] == pytest.approx(sum(r["opp"] for r in rs))
  assert out["fill"] == pytest.approx(sum(r["n_ok"] for r in rs) / sum(r["n_buy"] for r in rs))
  assert 0 < out["fill"] <= 1
  assert out["ret_buy"] == 0 and out["ret_sup"] == 1


# 같은 rep면 같은 결과, 다른 rep면 다른 결과
def test_rollout_reproducible():
  a, b, c = (rollout(CFG, Policy(CFG), rep) for rep in (0, 0, 1))
  assert a == b
  assert a["profit"] != c["profit"]


# 잉여 0이면 재참여 확률 = ret_p0, 잉여가 클수록 커지고, 잉여율 기준이라 잉여와 제안 금액을 같은 배수로 키우면 같다
@pytest.mark.parametrize("kind", ["buy", "sup"])
def test_logistic_response(kind):
  f = Logistic(CFG)
  assert f.prob(kind, 0.0, 400.0) == pytest.approx(CFG.ret_p0)
  p = f.prob(kind, np.array([0.0, 10.0, 20.0, 40.0]), 400.0)
  assert (np.diff(p) > 0).all() and p[-1] < 1
  assert f.prob(kind, 20.0, 400.0) == pytest.approx(f.prob(kind, 60.0, 1200.0))
  assert f.prob(kind, f.rate(kind, 0.7), 1.0) == pytest.approx(0.7)
