"""규칙 기반 잉여배분 정책 (주 비교 기준, 경험 수집·시뮬레이션 기준치의 후속 정책).
같은 MILP에서 잉여를 마진의 고정 비율(주문자 몫, 공급자 몫)로 묶고 유지 가치는 0이다.
몫은 주문자 × 공급자 몫 격자에서 기준 몫 sh_ref보다 튜닝용 반복(공통 난수)의 누적 이윤이 유의하게 큰 후보가 있을 때만 그 후보로 바꾼다.
"""
from dataclasses import replace
import numpy as np
from match.milp_solve import Policy
from match.interface import rollout


# 몫 split = (주문자 몫, 공급자 몫)의 규칙 기반 정책 (None이면 Cfg의 sh_buy, sh_sup)
def rule(cfg, split=None):
  return Policy(cfg, split=split or (cfg.sh_buy, cfg.sh_sup))


# 격자의 몫 후보별 튜닝용 반복(seed = tune_seed, n_tune회)의 누적 이윤 배열 (resp는 평가용 반응 함수)
def scores(cfg, resp=None):
  tc = replace(cfg, seed=cfg.tune_seed)
  grid = [(b, s) for b in cfg.sh_buy_grid for s in cfg.sh_sup_grid]
  return {sp: np.array([rollout(tc, rule(tc, sp), rep, resp)["profit"] for rep in range(cfg.n_tune)]) for sp in grid}


# 최선 몫: 평균이 가장 큰 후보가 기준 몫 sh_ref보다 짝지은 차이의 95% 신뢰구간으로 유의하게 크면 그 후보, 아니면 sh_ref
# sc를 주면 그 점수로 고른다 (튜닝 점수를 저장할 때 같은 점수를 쓰려고)
def best(cfg, resp=None, sc=None):
  sc = sc or scores(cfg, resp)
  sp = max(sc, key=lambda k: sc[k].mean())
  d = sc[sp] - sc[cfg.sh_ref]
  return sp if d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)) > 0 else cfg.sh_ref
