"""반응 함수 기울기 보정.
기준 잉여율(기준 몫 sh_ref로 마진의 30%를 받을 때, 잉여를 받은 참여자의 평균 잉여율)에서 재참여율이 ret_ss가 되도록 b_buy, b_sup를 정한다.
기준 잉여율이 기울기(남는 참여자 구성)에 따라 달라지므로 b × 기준 잉여율(b) = logit(ret_ss) − logit(ret_p0)을 이분탐색으로 푼다.
"""
from dataclasses import replace
import numpy as np
from env.platform import Env, worth
from match.interface import rollout
from match.milp_solve import Policy
from bench.rule_split import rule


# 몫 split(None이면 Cfg의 몫)의 규칙 기반 튜닝용 반복 평균 재참여율 (주문자, 공급자)
def ret_rate(cfg, split=None):
  tc = replace(cfg, seed=cfg.tune_seed)
  outs = [rollout(tc, Policy(tc, split=split) if split else rule(tc), rep) for rep in range(cfg.n_tune)]
  return np.mean([o["ret_buy"] for o in outs]), np.mean([o["ret_sup"] for o in outs])


# 기준 잉여율: 기준 몫 sh_ref로 운영한 튜닝용 반복에서 잉여를 받은 참여자의 잉여율(배분 잉여 / 제안 금액) 평균 (주문자, 공급자)
def ref_rate(cfg):
  tc = replace(cfg, seed=cfg.tune_seed)
  rb, rs = [], []
  for rep in range(cfg.n_tune):
    env, pol = Env(tc, rep), Policy(tc, split=cfg.sh_ref)
    env.reset()
    for _ in range(tc.T):
      wb, ws = worth(env.o)
      r = env.step(pol.act(env.o))
      rb += list(r["sb"][r["sb"] > 0] / wb[r["sb"] > 0])
      rs += list(r["ss"][r["ss"] > 0] / ws[r["ss"] > 0])
  return float(np.mean(rb)), float(np.mean(rs))


# 종류 kind의 기울기를 [0, cal_hi]에서 이분탐색: 그 기울기로 잰 기준 잉여율에서 재참여율이 ret_ss가 되게 (다른 종류 기울기는 고정)
def fit(cfg, kind):
  k = np.log(cfg.ret_ss / (1 - cfg.ret_ss)) - np.log(cfg.ret_p0 / (1 - cfg.ret_p0))
  lo, hi, key = 0.0, cfg.cal_hi, "b_" + kind
  for _ in range(cfg.cal_iter):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if mid * ref_rate(replace(cfg, **{key: mid}))[kind == "sup"] < k else (lo, mid)
  return replace(cfg, **{key: (lo + hi) / 2})


# 보정: 두 기울기를 cal_init에서 시작해 주문자·공급자를 번갈아 이분탐색 (cal_alt회 교대)
def calibrate(cfg):
  cfg = replace(cfg, b_buy=cfg.cal_init, b_sup=cfg.cal_init)
  for _ in range(cfg.cal_alt):
    cfg = fit(fit(cfg, "buy"), "sup")
  return cfg


if __name__ == "__main__":
  from utils.params import Cfg
  from env.response import Logistic
  cfg = calibrate(Cfg())
  rb, rs = ref_rate(cfg)
  f = Logistic(cfg)
  ob, os_ = ret_rate(cfg, cfg.sh_ref)
  print(f"final: b_buy={cfg.b_buy!r} b_sup={cfg.b_sup!r} ref_rate=({rb:.5f}, {rs:.5f}) "
        f"p=({f.prob('buy', rb, 1.0):.4f}, {f.prob('sup', rs, 1.0):.4f}) ref-share operating retention=({ob:.4f}, {os_:.4f})")
