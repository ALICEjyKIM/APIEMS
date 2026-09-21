"""환경 정산, 재참여 전이, rollout, 반응 함수 테스트.
손으로 계산 가능한 작은 시장에서 주문 성립 조건, 분할 공급, 용량·잉여 검사, 이윤 구성요소를 확인한다.
재참여 전이는 판정 규칙(ret_u < p)과 공통 난수, 공급자 자리 채우기를, rollout은 성과 지표 집계를 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Obs, Env, settle
from env.response import Logistic
from utils.common import ret_u
from utils.arrivals import perturb
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


RULE = (CFG.sh_buy, CFG.sh_sup)


# 재참여 판정: 기간 t의 모든 주문자·공급자는 ret_u < p(배분 잉여 / 제안 금액)일 때만 남는다 (거절된 주문자 포함)
def test_stay_rule():
  env, pol, f = Env(CFG, 0), Policy(CFG, split=RULE), Logistic(CFG)
  env.reset()
  n_stay = 0
  for t in range(8):
    o = env.o
    r = env.step(pol.act(o))
    assert r["pb"] == pytest.approx(f.prob("buy", r["sb"], (o.p * o.q).sum(1)))
    assert r["ps"] == pytest.approx(f.prob("sup", r["ss"], (o.c * o.cap).sum(1)))
    sb = {i for i, p in zip(o.bid, r["pb"]) if ret_u(CFG, 0, "buy", i, t) < p}
    ss = {i for i, p in zip(o.sid, r["ps"]) if ret_u(CFG, 0, "sup", i, t) < p}
    assert set(o.bid) & set(env.o.bid) == sb and set(o.sid) & set(env.o.sid) == ss
    assert r["stay_buy"] == len(sb) and r["stay_sup"] == len(ss)
    n_stay += len(sb)
  assert n_stay > 0


# 재참여 주문자는 프로필의 품목 구성 그대로, 프로필에 변동을 더한 주문을 낸다
def test_returning_buyer_order():
  env, pol = Env(CFG, 0), Policy(CFG, split=RULE)
  env.reset()
  seen = 0
  for t in range(8):
    prev = set(env.o.bid)
    env.step(pol.act(env.o))
    for row, i in enumerate(env.o.bid):
      if i in prev:
        seen += 1
        q = perturb(CFG, 0, "buy", i, env.t, env.buys[i])
        assert np.array_equal(env.o.q[row], q.qty) and np.array_equal(env.o.p[row], q.price)
        assert np.array_equal(env.o.q[row] > 0, env.buys[i].items)
  assert seen > 0


# 떠난 공급자 자리는 같은 품목·공급가능량의 새 공급자가 새 번호(t × n_sup + j)로 채운다 (p_sup_new = 1)
def test_supplier_refill():
  cfg = replace(CFG, p_sup_new=1.0)
  env, pol = Env(cfg, 0), Policy(cfg)
  env.reset()
  left = 0
  for _ in range(5):
    o = env.o
    env.step(pol.act(o))
    assert len(env.o.sid) == cfg.n_sup
    for j, (i, new) in enumerate(zip(o.sid, env.o.sid)):
      if i != new:
        left += 1
        assert new == env.t * cfg.n_sup + j
        assert np.array_equal(env.o.cap[j], env.inst.sup_cap[j])
  assert left > 0


# 공급자가 모두 떠나 빈 기간도 정산된다 (p_sup_new = 0)
def test_all_suppliers_gone():
  cfg = replace(CFG, p_sup_new=0.0)
  env, pol = Env(cfg, 0), Policy(cfg)
  env.reset()
  for _ in range(cfg.T):
    env.step(pol.act(env.o))
  assert len(env.o.sid) == 0
  r = env.step(pol.act(env.o))
  assert r["n_ok"] == 0 and r["profit"] == 0


# 공통 난수: 결정이 달라도 신규 주문자와, 양쪽에 모두 남은 참여자의 다음 주문·공급은 같다
def test_transition_common_random():
  a, b = Env(CFG, 0), Env(CFG, 0)
  a.reset(), b.reset()
  pa, pb = Policy(CFG), Policy(CFG, split=RULE)
  both = 0
  for _ in range(8):
    prev_a, prev_b = set(a.o.bid), set(b.o.bid)
    a.step(pa.act(a.o)), b.step(pb.act(b.o))
    assert set(a.o.bid) - prev_a == set(b.o.bid) - prev_b
    ra, rb = dict(zip(a.o.bid, range(len(a.o.bid)))), dict(zip(b.o.bid, range(len(b.o.bid))))
    for i in set(ra) & set(rb):
      both += 1
      assert np.array_equal(a.o.q[ra[i]], b.o.q[rb[i]]) and np.array_equal(a.o.p[ra[i]], b.o.p[rb[i]])
  assert both > 0


# 근시안은 모두에게 잉여 0을 주므로 재참여율 = ret_p0 (표본 오차 이내)
def test_myopic_retention_is_p0():
  outs = [rollout(CFG, Policy(CFG), rep) for rep in range(5)]
  assert np.mean([o["ret_buy"] for o in outs]) == pytest.approx(CFG.ret_p0, abs=0.03)
  assert np.mean([o["ret_sup"] for o in outs]) == pytest.approx(CFG.ret_p0, abs=0.05)


# rollout: 누적 이윤 = 수익 − 배송비, 주문 충족률 = 성립 주문 / 전체 주문, 유지율 = 남은 참여자 / 참여자, 평균 활동 참여자 수
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rollout_metrics(k):
  cfg = replace(CFG, items_per_order=k)
  env, pol = Env(cfg, 0), Policy(cfg, split=RULE)
  env.reset()
  pol.reset(0)
  rs = [env.step(pol.act(env.o)) for _ in range(cfg.T)]
  out = rollout(cfg, Policy(cfg, split=RULE), 0)
  assert out["profit"] == pytest.approx(sum(r["profit"] for r in rs))
  assert out["profit"] == pytest.approx(out["rev"] - out["ship"])
  assert out["opp"] == pytest.approx(sum(r["opp"] for r in rs))
  assert out["fill"] == pytest.approx(sum(r["n_ok"] for r in rs) / sum(r["n_buy"] for r in rs))
  assert out["ret_buy"] == pytest.approx(sum(r["stay_buy"] for r in rs) / sum(r["n_buy"] for r in rs))
  assert out["ret_sup"] == pytest.approx(sum(r["stay_sup"] for r in rs) / sum(r["n_sup"] for r in rs))
  assert out["act_buy"] == pytest.approx(sum(r["n_buy"] for r in rs) / cfg.T)
  assert out["act_sup"] == pytest.approx(sum(r["n_sup"] for r in rs) / cfg.T)
  assert 0 < out["fill"] <= 1 and 0 < out["ret_buy"] < 1 and 0 < out["ret_sup"] <= 1


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
