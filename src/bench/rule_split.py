"""규칙 기반 잉여배분 정책 (주 비교 기준, 경험 수집·시뮬레이션 기준치의 후속 정책).
같은 MILP에서 잉여를 마진의 고정 비율(주문자 sh_buy, 공급자 몫)로 묶고 유지 가치는 0이다.
공급자 몫은 튜닝용 seed 반복(공통 난수)에서 기본값보다 누적 이윤이 유의하게 큰 후보가 있을 때만 그 후보로 바꾼다.
"""
from dataclasses import replace
import numpy as np
from match.milp_solve import Policy
from match.interface import rollout


# 공급자 몫 s의 규칙 기반 정책 (None이면 Cfg.sh_sup)
def rule(cfg, s=None):
  return Policy(cfg, split=(cfg.sh_buy, cfg.sh_sup if s is None else s))


# 공급자 몫 후보별 튜닝용 반복(seed = tune_seed, n_tune회)의 누적 이윤 배열 (resp는 평가용 반응 함수)
def scores(cfg, resp=None):
  tc = replace(cfg, seed=cfg.tune_seed)
  return {s: np.array([rollout(tc, rule(tc, s), rep, resp)["profit"] for rep in range(cfg.n_tune)]) for s in cfg.sh_sup_grid}


# 최선 공급자 몫: 평균이 가장 큰 후보가 기본값 Cfg.sh_sup보다 짝지은 차이의 95% 신뢰구간으로 유의하게 크면 그 후보, 아니면 기본값
def best(cfg, resp=None):
  sc = scores(cfg, resp)
  s = max(sc, key=lambda k: sc[k].mean())
  d = sc[s] - sc[cfg.sh_sup]
  return s if d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)) > 0 else cfg.sh_sup
