"""반응 함수 기울기 보정.
기준 잉여율(기준 몫 sh_ref, 즉 마진의 30%를 받을 때 잉여를 받은 참여자의 평균 잉여율)에서 재참여율이 ret_ss가 되도록 b_buy, b_sup를 닫힌 식으로 정한다.
기준 잉여율은 튜닝용 반복에서 재며, 측정이 기울기(참여자 구성)에 조금 의존하므로 측정 → 설정을 cal_rounds회 반복한다.
"""
from dataclasses import replace
import numpy as np
from env.platform import Env, worth
from match.interface import rollout
from bench.rule_split import rule
from match.milp_solve import Policy


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
  return np.mean(rb), np.mean(rs)


# 보정: 기울기 b = (logit(ret_ss) − logit(ret_p0)) / 기준 잉여율. 기준 잉여율을 새 기울기로 다시 재며 cal_rounds회 반복한다
def calibrate(cfg):
  k = np.log(cfg.ret_ss / (1 - cfg.ret_ss)) - np.log(cfg.ret_p0 / (1 - cfg.ret_p0))
  for i in range(cfg.cal_rounds):
    rb, rs = ref_rate(cfg)
    cfg = replace(cfg, b_buy=k / rb, b_sup=k / rs)
    print(f"round {i}: ref_rate=({rb:.5f}, {rs:.5f}) -> b_buy={cfg.b_buy!r} b_sup={cfg.b_sup!r}", flush=True)
  return cfg


if __name__ == "__main__":
  from utils.params import Cfg
  cfg = calibrate(Cfg())
  rb, rs = ret_rate(cfg, cfg.sh_ref)
  print(f"final: b_buy={cfg.b_buy!r} b_sup={cfg.b_sup!r}; 기준 몫 {cfg.sh_ref} 운영 재참여율=({rb:.4f}, {rs:.4f})")
