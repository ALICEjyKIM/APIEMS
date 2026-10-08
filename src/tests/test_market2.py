"""조달 시장 v2 테스트.
시장 생성(품목별 대체 공급자 수, 연결 구조), 공급자 교체와 onboarding 지연, 주문자 프로필 지속, 경험 의존 재참여,
공통 난수, MILP의 성립·배정 경험 항을 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg, PROC15
from utils.market2 import make2, candidate, demand
from env.platform import Env, Env2, Obs, worth
from env.response import Retention
from match.milp_solve import Policy
from bench.rule_split import rule

CFG = replace(Cfg(), **PROC15)


# het = 0이면 모든 품목이 n_alt개 공급자, het > 0이면 품목마다 다르다. 연결 구조는 목표 수와 자리 품목 수를 정확히 맞춘다
@pytest.mark.parametrize("het", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("conc", [0.0, 0.5, 1.0])
def test_network(het, conc):
  cfg = replace(CFG, het=het, conc=conc)
  for rep in range(5):
    mk = make2(cfg, rep)
    items = np.array([s.items for s in mk.sups])
    assert (items.sum(0) == mk.n_tgt).all() and (items.sum(1) == mk.deg).all()
    assert mk.n_tgt.sum() == cfg.n_items * cfg.n_alt
    assert (mk.n_tgt == cfg.n_alt).all() if het == 0 else mk.n_tgt.min() < mk.n_tgt.max()
    cap = np.array([s.qty for s in mk.sups])
    assert ((cap > 0) == items).all()


# 품목 명목 용량은 정책과 무관한 cover × 명목 수요이고, 기준가는 legacy와 같은 첫 난수다
def test_capacity_and_base():
  from utils.instance import make
  mk = make2(CFG, 0)
  assert np.allclose(mk.K, CFG.cover * demand(CFG))
  assert np.allclose(mk.base, make(replace(CFG, market="legacy"), 0).base)


# 공급자를 빼면 그 자리는 onboard 기간 동안 비고 후보는 pend·wait로만 보이며, 이후 새 프로필(새 ID)로 채워진다
def test_onboarding():
  env, pol = Env(CFG, 0), Policy(CFG)
  env.reset()
  j, old = 0, env.offs[0]
  env.drop("sup", 0)
  assert j not in env.offs and len(env.o.pend) == 1 and env.o.wait.tolist() == [CFG.onboard]
  for t in range(1, CFG.onboard + 1):
    env.step(pol.act(env.o))
    if t < CFG.onboard and j in env.pipe:
      assert j not in env.offs and env.o.wait[sorted(env.pipe).index(j)] == CFG.onboard - t
  assert j in env.offs and env.sups[j][0] == CFG.onboard * CFG.n_sup + j
  new = env.sups[j][1]
  assert not (np.array_equal(new.items, old.items) and np.array_equal(new.qty, old.qty) and np.allclose(new.price, old.price))
  assert new.items.sum() == old.items.sum() == env.inst.deg[j]


# 같은 (반복, 기간, 자리)의 후보는 같고, 기간이나 자리가 다르면 다르다 (공통 난수)
def test_candidate_crn():
  mk = make2(CFG, 0)
  a, b = candidate(CFG, mk, 0, 3, 5), candidate(CFG, mk, 0, 3, 5)
  assert np.array_equal(a.qty, b.qty) and np.allclose(a.price, b.price)
  c = candidate(CFG, mk, 0, 4, 5)
  assert not (np.array_equal(a.qty, c.qty) and np.allclose(a.price, c.price))


# 정책이 달라도 같은 기간의 신규 주문자(ID·주문)는 같다
def test_arrivals_crn():
  ea, eb = Env(CFG, 1), Env(CFG, 1)
  ea.reset(), eb.reset()
  pa, pb = Policy(CFG), rule(CFG)
  for _ in range(8):
    na, nb = ea.nb, eb.nb
    ea.step(pa.act(ea.o)), eb.step(pb.act(eb.o))
    assert ea.nb - na == eb.nb - nb
    new = [i for i in ea.o.bid if i >= na]
    qa = {i: q for i, q in zip(ea.o.bid, ea.o.q)}
    qb = {i: q for i, q in zip(eb.o.bid, eb.o.q)}
    assert all(np.array_equal(qa[i], qb[i]) for i in new)


# 남은 주문자는 처음 주문한 품목 묶음을 그대로 반복한다 (수량·가격만 작게 변한다)
def test_buyer_persistence():
  env, pol = Env(CFG, 2), rule(CFG)
  env.reset()
  first = {i: q > 0 for i, q in zip(env.o.bid, env.o.q)}
  seen = 0
  for _ in range(10):
    env.step(pol.act(env.o))
    for i, q in zip(env.o.bid, env.o.q):
      if i in first:
        assert np.array_equal(q > 0, first[i])
        seen += 1
      else:
        first[i] = q > 0
  assert seen > 0


# 반응 함수 세 점과 경험 차이
def test_retention_anchors():
  f = Retention(CFG)
  assert f.prob("buy", 0.0, 1.0, 0.0) == pytest.approx(CFG.r2_p0)
  assert f.prob("sup", 0.0, 1.0, 1.0) == pytest.approx(CFG.r2_p1)
  assert f.prob("buy", CFG.r2_rate, 1.0, 1.0) == pytest.approx(CFG.r2_p2)
  assert f.gap("sup") == pytest.approx(CFG.r2_p1 - CFG.r2_p0)
  assert f.prob("sup", f.rate("sup", 0.9), 1.0) == pytest.approx(0.9)


# 환경의 재참여확률은 경험(주문 성립, 거래 배정)과 잉여율로 정해진다: 경험이 없고 잉여 0이면 r2_p0, 있으면 r2_p1 이상
def test_env_retention_uses_experience():
  env, pol = Env(CFG, 0), rule(CFG)
  env.reset()
  hit = {"b0": 0, "b1": 0, "s0": 0, "s1": 0}
  for _ in range(6):
    o = env.o
    r = env.step(pol.act(o))
    wb, ws = worth(o)
    assert np.allclose(r["pb"], Retention(CFG).prob("buy", r["sb"], wb, r["ok"]))
    assert np.allclose(r["ps"], Retention(CFG).prob("sup", r["ss"], ws, r["alloc"]))
    assert np.allclose(r["pb"][~r["ok"]], CFG.r2_p0) and (r["pb"][r["ok"]] >= CFG.r2_p1 - 1e-9).all()
    assert np.allclose(r["ps"][~r["alloc"]], CFG.r2_p0) and (r["ps"][r["alloc"]] >= CFG.r2_p1 - 1e-9).all()
    hit["b0"] += (~r["ok"]).sum()
    hit["b1"] += r["ok"].sum()
    hit["s0"] += (~r["alloc"]).sum()
    hit["s1"] += r["alloc"].sum()
    assert r["x"].shape == (len(o.bid), len(o.sid), CFG.n_items) and len(r["kept_b"]) == len(o.bid) and len(r["kept_s"]) == len(o.sid)
  assert min(hit.values()) > 0


# MILP 주문자 성립 항: 손해 5인 주문은 근시안이면 거절하고, 주문자 유지 가치 c가 c × (r2_p1 − r2_p0) > 5이면 수락한다
@pytest.mark.parametrize("c,acc", [(0.0, False), (10.0, False), (100.0, True)])
def test_milp_buyer_fulfillment(c, acc):
  o = Obs(0, np.array([[10], [10]]), np.array([[20.0], [12.0]]), np.array([[20]]), np.array([[10.0]]), [0, 1], [0])
  pol = Policy(CFG, coef=lambda o: (np.array([0.0, c]), np.zeros(1)))
  x, u, v = pol.act(o)
  assert x[0].sum() == 10 and (x[1].sum() == 10) == acc


# MILP 공급자 배정 항: 비싼 공급자(단위 마진 2 작음)에게 1개라도 배정하면 c × (r2_p1 − r2_p0)를 얻는다
@pytest.mark.parametrize("c,alloc", [(0.0, False), (5.0, False), (100.0, True)])
def test_milp_supplier_allocation(c, alloc):
  o = Obs(0, np.array([[10]]), np.array([[20.0]]), np.array([[10], [10]]), np.array([[10.0], [12.0]]), [0], [0, 1])
  pol = Policy(CFG, coef=lambda o: (np.zeros(1), np.array([0.0, c])))
  x, u, v = pol.act(o)
  assert x.sum() == 10 and (x[0, 1].sum() > 0) == alloc


# 유지 가치가 있어도 매 기간 최적해이고 결정이 환경 검사를 통과한다. 잉여가 0인 기간은 목적함수 = 이윤 + Σ c × 환경 재참여확률
@pytest.mark.parametrize("cv", [0.0, 30.0])
def test_v2_milp_rollout(cv):
  env = Env(CFG, 3)
  f = Retention(CFG)
  pol = Policy(CFG, split=(CFG.sh_buy, None), coef=lambda o: (np.zeros(len(o.bid)), np.full(len(o.sid), cv)))
  env.reset()
  for _ in range(CFG.T):
    o = env.o
    d = pol.act(o)
    r = env.step(d)
    assert r["profit"] >= -CFG.tol
    if r["ss"].sum() == 0:
      ws = worth(o)[1]
      assert pol.obj == pytest.approx(r["profit"] + cv * f.prob("sup", 0.0, ws, r["alloc"]).sum(), abs=1e-4)
