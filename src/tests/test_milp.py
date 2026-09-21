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
from tests.test_env import split_obs, bundle_obs

CFG = Cfg()
SH = 1 - CFG.sh_buy - CFG.sh_sup


# 한 공급자 용량(6)으로 못 채우는 주문(10)을 두 공급자가 나눠 채운다: 싼 공급자 6, 비싼 공급자 4
def test_split_supply():
  pol, o = Policy(CFG), split_obs()
  x = pol.act(o)
  assert x.tolist() == [[[6, 0], [4, 0]]]
  assert pol.obj == pytest.approx(SH * (10 * 6 + 8 * 4) - CFG.f_ship)
  assert settle(CFG, o, x)["ok"].tolist() == [True]


# 묶음의 한 품목 용량이 모자라면 주문 전체를 거절하고 아무것도 배정하지 않는다
def test_bundle_rejected_if_any_item_short():
  o = bundle_obs()
  o.cap = np.array([[5, 4]])
  x = Policy(CFG).act(o)
  assert (x == 0).all()


# 플랫폼 몫이 배송 고정비보다 작은 주문은 거절하고, 크면 수락한다
def test_fixed_ship_cost_threshold():
  o = split_obs()
  o.cap = np.array([[10, 0], [0, 0]])
  o.p = np.array([[10.0 + CFG.f_ship / (10 * SH) - 0.1, 0]])
  assert (Policy(CFG).act(o) == 0).all()
  o.p = np.array([[10.0 + CFG.f_ship / (10 * SH) + 0.1, 0]])
  assert Policy(CFG).act(o).tolist() == [[[10, 0], [0, 0]]]


# 주문자가 없는 기간도 최적해(빈 배정)
def test_no_buyers():
  o = Obs(0, np.zeros((0, 2), int), np.zeros((0, 2)), np.array([[6, 0]]), np.array([[10.0, 0]]), [], [0])
  x = Policy(CFG).act(o)
  assert x.shape == (0, 1, 2) and np.issubdtype(x.dtype, np.integer)


# 모든 기간에서 최적해이고, MILP 목적함수 값 = 환경 정산 이윤 (시장 조건 전체)
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("conc", [0.0, 1.0])
def test_milp_matches_env(k, conc):
  cfg = replace(CFG, items_per_order=k, conc=conc)
  env, pol = Env(cfg, 0), Policy(cfg)
  env.reset()
  pol.reset(0)
  for _ in range(cfg.T):
    r = env.step(pol.act(env.o))
    assert r["profit"] == pytest.approx(pol.obj, abs=1e-6)
    assert r["profit"] >= 0
