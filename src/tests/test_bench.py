"""비교군 테스트 (근시안, 규칙 기반).
근시안은 잉여를 주지 않고, 규칙 기반은 잉여를 마진의 고정 비율로 준다.
규칙 기반 튜닝은 튜닝용 seed의 공통 반복으로 후보를 비교해 가장 좋은 공급자 몫을 고른다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Env
from match.interface import rollout
from bench.myopic import myopic
from bench.rule_split import rule, scores, best
from bench.tune import ret_rate, ref_rate
from env.response import Logistic

CFG = Cfg()


# 근시안: 매 기간 잉여 0 (플랫폼이 마진 − 배송비를 모두 가져감)
def test_myopic_zero_surplus():
  env, pol = Env(CFG, 0), myopic(CFG)
  env.reset()
  for _ in range(10):
    r = env.step(pol.act(env.o))
    assert np.allclose(r["sb"], 0) and np.allclose(r["ss"], 0)


# 규칙 기반: 주문자 잉여 = sh_buy × 주문 마진, 공급자 잉여 = 공급자 몫 × 공급자 마진
@pytest.mark.parametrize("s", [0.1, 0.4])
def test_rule_fixed_split(s):
  env, pol = Env(CFG, 0), rule(CFG, s)
  env.reset()
  for _ in range(10):
    o = env.o
    x, u, v = pol.act(o)
    m = (o.p[:, None] - o.c[None]) * x
    assert u == pytest.approx(CFG.sh_buy * m.sum((1, 2)), abs=1e-6)
    assert v == pytest.approx(s * m.sum((0, 2)), abs=1e-6)
    env.step((x, u, v))


# 두 비교군이 끝까지 돌고, 잉여를 주는 규칙 기반이 근시안보다 재참여율이 높다 (공통 난수, 같은 반복)
def test_baselines_run():
  a = [rollout(CFG, myopic(CFG), rep) for rep in range(3)]
  b = [rollout(CFG, rule(CFG), rep) for rep in range(3)]
  for o in a + b:
    assert o["profit"] > 0 and 0 < o["fill"] <= 1
  for k in ("ret_buy", "ret_sup"):
    assert np.mean([o[k] for o in b]) > np.mean([o[k] for o in a])


# 튜닝: 후보별 튜닝용 seed의 같은 반복으로 누적 이윤을 재고, 기본값보다 유의하게 큰 후보가 있을 때만 바꾼다
# 공급자 몫 0.7은 플랫폼 몫이 0이라 주문을 모두 거절해 이윤 0: 기본값 0.3이면 유지, 기본값 0.7이면 0.3으로 바뀐다
def test_best_split():
  cfg = replace(CFG, T=8, n_tune=4, sh_sup_grid=(0.3, 0.7))
  sc = scores(cfg)
  assert set(sc) == set(cfg.sh_sup_grid) and len(sc[0.3]) == cfg.n_tune
  tc = replace(cfg, seed=cfg.tune_seed)
  assert sc[0.3] == pytest.approx([rollout(tc, rule(tc, 0.3), rep)["profit"] for rep in range(cfg.n_tune)])
  assert (sc[0.7] == 0).all() and (sc[0.3] > 0).all()
  assert best(cfg) == 0.3
  assert best(replace(cfg, sh_sup=0.7)) == 0.3
  assert not np.array_equal(sc[0.3], scores(replace(cfg, tune_seed=cfg.tune_seed + 1))[0.3])


# 보정: Cfg 기울기는 기준 잉여율(기준 몫 sh_ref로 운영해 잉여를 받은 참여자의 평균 잉여율)에서 재참여율 ret_ss를 준다
def test_calibrated_reference():
  rb, rs = ref_rate(CFG)
  f = Logistic(CFG)
  assert f.prob("buy", rb, 1.0) == pytest.approx(CFG.ret_ss, abs=0.02)
  assert f.prob("sup", rs, 1.0) == pytest.approx(CFG.ret_ss, abs=0.02)


# 이분탐색은 기울기를 올리면 재참여율이 오르는 성질을 쓴다 (소규모 설정으로 확인)
def test_slope_monotone():
  cfg = replace(CFG, T=8, n_tune=2)
  lo, hi = ret_rate(replace(cfg, b_buy=5.0, b_sup=5.0)), ret_rate(replace(cfg, b_buy=100.0, b_sup=100.0))
  assert lo[0] < hi[0] and lo[1] < hi[1]
