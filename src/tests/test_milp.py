"""MILP 정책 테스트 (슬라이스 1: 주문 수락과 품목별 공급자·수량 배정).
손으로 푼 작은 시장에서 분할 공급, 묶음 성립 조건, 배송 고정비에 따른 거절을 확인한다.
매 기간 MILP가 최적해이고 목적함수 값이 환경 정산 이윤과 같은지 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Obs, Env, settle
from match.milp_solve import Policy
from env.response import Logistic
from utils.pwl import points
from tests.test_env import split_obs, bundle_obs

CFG = Cfg()
SH = 1 - CFG.sh_buy - CFG.sh_sup


RULE = (CFG.sh_buy, CFG.sh_sup)


# 규칙 기반: 한 공급자 용량(6)으로 못 채우는 주문(10)을 두 공급자가 나눠 채우고(싼 공급자 6, 비싼 공급자 4) 잉여는 고정 비율
def test_split_supply():
  pol, o = Policy(CFG, split=RULE), split_obs()
  x, u, v = pol.act(o)
  assert x.tolist() == [[[6, 0], [4, 0]]]
  assert u == pytest.approx([CFG.sh_buy * 92]) and v == pytest.approx([CFG.sh_sup * 60, CFG.sh_sup * 32])
  assert pol.obj == pytest.approx(SH * (10 * 6 + 8 * 4) - CFG.f_ship)
  assert settle(CFG, o, (x, u, v))["ok"].tolist() == [True]


# 근시안(유지 가치 0, 잉여 자유): 같은 배정에 잉여를 주지 않고 마진 − 배송비를 모두 가져간다
def test_myopic_takes_all():
  pol, o = Policy(CFG), split_obs()
  x, u, v = pol.act(o)
  assert x.tolist() == [[[6, 0], [4, 0]]]
  assert (u == 0).all() and (v == 0).all()
  assert pol.obj == pytest.approx(92 - CFG.f_ship)


# 묶음의 한 품목 용량이 모자라면 주문 전체를 거절하고 아무것도 배정하지 않는다
@pytest.mark.parametrize("split", [None, RULE])
def test_bundle_rejected_if_any_item_short(split):
  o = bundle_obs()
  o.cap = np.array([[5, 4]])
  x, u, v = Policy(CFG, split=split).act(o)
  assert (x == 0).all() and (u == 0).all() and (v == 0).all()


# 플랫폼 몫이 배송 고정비보다 작은 주문은 거절하고, 크면 수락한다 (규칙 기반은 몫 SH, 근시안은 마진 전체)
@pytest.mark.parametrize("split,sh", [(RULE, SH), (None, 1.0)])
def test_fixed_ship_cost_threshold(split, sh):
  o = split_obs()
  o.cap = np.array([[10, 0], [0, 0]])
  o.p = np.array([[10.0 + CFG.f_ship / (10 * sh) - 0.1, 0]])
  assert (Policy(CFG, split=split).act(o)[0] == 0).all()
  o.p = np.array([[10.0 + CFG.f_ship / (10 * sh) + 0.1, 0]])
  assert Policy(CFG, split=split).act(o)[0].tolist() == [[[10, 0], [0, 0]]]


# 주문자가 없는 기간도 최적해(빈 배정)
def test_no_buyers():
  o = Obs(0, np.zeros((0, 2), int), np.zeros((0, 2)), np.array([[6, 0]]), np.array([[10.0, 0]]), [], [0])
  x, u, v = Policy(CFG).act(o)
  assert x.shape == (0, 1, 2) and np.issubdtype(x.dtype, np.integer) and u.shape == (0,) and v.shape == (1,)


# 모든 기간에서 최적해이고, MILP 목적함수 값 = 환경 정산 이윤 (시장 조건 전체, 근시안·규칙 기반)
@pytest.mark.parametrize("split", [None, RULE])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("conc", [0.0, 1.0])
def test_milp_matches_env(k, conc, split):
  cfg = replace(CFG, items_per_order=k, conc=conc)
  env, pol = Env(cfg, 0), Policy(cfg, split=split)
  env.reset()
  pol.reset(0)
  for _ in range(cfg.T):
    r = env.step(pol.act(env.o))
    assert r["profit"] == pytest.approx(pol.obj, abs=1e-6)
    assert r["profit"] >= 0


# 구간선형 근사: 구간 5개, 잉여율 0에서 ret_p0, 끝점은 pwl_rmax, 잉여율 [0, pwl_rmax]에서 원래 곡선과의 최대 오차 ≤ 0.03
@pytest.mark.parametrize("b", [10.0, 30.0, 100.0])
@pytest.mark.parametrize("kind", ["buy", "sup"])
def test_pwl_close(b, kind):
  cfg = replace(CFG, b_buy=b, b_sup=b)
  f = Logistic(cfg)
  r, p = points(cfg, f, kind)
  assert len(r) - 1 == 5 and r[0] == 0 and r[-1] == cfg.pwl_rmax and (np.diff(r) > 0).all()
  assert p[0] == pytest.approx(cfg.ret_p0)
  g = np.linspace(0, cfg.pwl_rmax, 20001)
  assert np.abs(np.interp(g, r, p) - f.prob(kind, g, 1.0)).max() <= 0.03
