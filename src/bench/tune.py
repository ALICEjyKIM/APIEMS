"""반응 함수 기울기 보정 (이분탐색).
규칙 기반 정책(최선 공급자 몫)으로 튜닝용 반복을 돌렸을 때 주문자·공급자 재참여율이 ret_ss가 되도록 b_buy, b_sup를 맞춘다.
최선 공급자 몫이 보정에 따라 바뀔 수 있어 "보정 → 최선 몫 재튜닝"을 몫이 바뀌지 않을 때까지 반복한다.
"""
from dataclasses import replace
import numpy as np
from match.interface import rollout
from bench.rule_split import rule, best


# 규칙 기반(Cfg.sh_sup) 튜닝용 반복의 평균 재참여율 (주문자, 공급자)
def ret_rate(cfg):
  tc = replace(cfg, seed=cfg.tune_seed)
  outs = [rollout(tc, rule(tc), rep) for rep in range(cfg.n_tune)]
  return np.mean([o["ret_buy"] for o in outs]), np.mean([o["ret_sup"] for o in outs])


# 종류 kind의 기울기를 [0, cal_hi]에서 이분탐색해 재참여율을 ret_ss에 맞춘다 (다른 종류 기울기는 고정)
def fit(cfg, kind):
  lo, hi, key = 0.0, cfg.cal_hi, "b_" + kind
  for _ in range(cfg.cal_iter):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if ret_rate(replace(cfg, **{key: mid}))[kind == "sup"] < cfg.ret_ss else (lo, mid)
  return replace(cfg, **{key: (lo + hi) / 2})


# 보정: 기울기를 cal_init에서 시작해 교대 이분탐색 → 최선 공급자 몫 재튜닝, 몫이 바뀌지 않으면 끝
# cal_rounds회 안에 수렴하지 않으면 마지막 몫으로 기울기만 한 번 더 맞춰 (기울기, 몫)이 서로 맞는 설정을 돌려준다
def calibrate(cfg):
  cfg = replace(cfg, b_buy=cfg.cal_init, b_sup=cfg.cal_init)
  for k in range(cfg.cal_rounds + 1):
    for _ in range(cfg.cal_alt):
      cfg = fit(fit(cfg, "buy"), "sup")
    s = best(cfg) if k < cfg.cal_rounds else cfg.sh_sup
    rb, rs = ret_rate(cfg)
    print(f"round {k}: b_buy={cfg.b_buy!r} b_sup={cfg.b_sup!r} sh_sup={cfg.sh_sup} -> best {s}, ret=({rb:.4f}, {rs:.4f})", flush=True)
    if s == cfg.sh_sup:
      return cfg
    cfg = replace(cfg, sh_sup=s)


if __name__ == "__main__":
  from utils.params import Cfg
  cfg = calibrate(Cfg())
  rb, rs = ret_rate(cfg)
  print(f"final: b_buy={cfg.b_buy!r} b_sup={cfg.b_sup!r} sh_sup={cfg.sh_sup} ret=({rb:.4f}, {rs:.4f})")
