"""반응 함수 두 점 설정과 공급자 자리 점유율 측정.
반응: 잉여 0이면 ret_p0, 기준 잉여율 r_ref(기준 몫 sh_ref로 마진의 30%를 받을 때의 평균 잉여율)이면 ret_ref가 되는 로지스틱.
기준 잉여율과 점유율은 튜닝용 반복에서 기준 몫으로 운영해 한 번씩 재서 Cfg에 고정한다 (반복 보정·이분탐색 없음).
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


# 공급자 자리 점유율: 기준 몫 sh_ref로 운영한 튜닝용 반복의 평균 활동 공급자 수 / 공급자 자리 수
def occupancy(cfg):
  tc = replace(cfg, seed=cfg.tune_seed)
  return float(np.mean([rollout(tc, Policy(tc, split=cfg.sh_ref), rep)["act_sup"] for rep in range(cfg.n_tune)]) / cfg.n_sup)


# 두 점 기울기: b = (logit(ret_ref) − logit(ret_p0)) / r_ref
def slopes(cfg, rb, rs):
  k = np.log(cfg.ret_ref / (1 - cfg.ret_ref)) - np.log(cfg.ret_p0 / (1 - cfg.ret_p0))
  return replace(cfg, r_ref_buy=rb, r_ref_sup=rs, b_buy=float(k / rb), b_sup=float(k / rs))


# 한 번씩 측정: (1) 기준 잉여율을 재서 두 점 기울기를 정하고 (2) 그 반응으로 점유율을 재서 용량에 쓴다
def measure(cfg):
  cfg = slopes(cfg, *ref_rate(cfg))
  return replace(cfg, occ=occupancy(cfg))


if __name__ == "__main__":
  from utils.params import Cfg
  cfg = measure(replace(Cfg(), price_rank=False))  # Cfg의 기울기·점유율은 기존 가격 생성 방식에서 잰 값
  print(f"final: r_ref=({cfg.r_ref_buy!r}, {cfg.r_ref_sup!r}) b=({cfg.b_buy!r}, {cfg.b_sup!r}) occ={cfg.occ!r}")
