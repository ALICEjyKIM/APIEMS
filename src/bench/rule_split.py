"""규칙 기반 잉여배분 정책 (주 비교 기준, 경험 수집·시뮬레이션 기준치의 후속 정책).
같은 MILP에서 잉여를 마진의 고정 비율(주문자 sh_buy, 공급자 몫)로 묶고 유지 가치는 0이다.
공급자 몫은 튜닝용 seed 반복의 평균 누적 이윤으로 후보 중 최선을 고른다 (공통 난수, 시장 조건마다 다시).
"""
from dataclasses import replace
import numpy as np
from match.milp_solve import Policy
from match.interface import rollout


# 공급자 몫 s의 규칙 기반 정책 (None이면 Cfg.sh_sup)
def rule(cfg, s=None):
  return Policy(cfg, split=(cfg.sh_buy, cfg.sh_sup if s is None else s))


# 공급자 몫 후보별 튜닝용 반복(seed = tune_seed, n_tune회)의 평균 누적 이윤 (resp는 평가용 반응 함수)
def scores(cfg, resp=None):
  tc = replace(cfg, seed=cfg.tune_seed)
  return {s: np.mean([rollout(tc, rule(tc, s), rep, resp)["profit"] for rep in range(cfg.n_tune)]) for s in cfg.sh_sup_grid}


# 최선 공급자 몫
def best(cfg, resp=None):
  sc = scores(cfg, resp)
  return max(sc, key=sc.get)
