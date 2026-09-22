"""가치 근사 학습 데이터.
튜닝 seed(평가 seed와 분리)에서 후속 정책(규칙 기반, Cfg 몫)으로 돌린 rollout의 (시장 요약 지표, 이후 vf_H기간 이윤 합)을 모은다.
검증 분할(학습 반복의 뒤 vf_val 비율)은 모든 근사 방법이 같은 것을 쓴다.
"""
from dataclasses import replace
import numpy as np
from env.platform import Env
from bench.rule_split import rule
from utils.features import phi


# 튜닝 seed 규칙 기반 rollout vf_reps반복의 (지표 행렬, 이후 vf_H기간 이윤 합, 반복 번호). 창이 T 안에 드는 t ≤ T − vf_H만
def collect(cfg):
  tc = replace(cfg, seed=cfg.tune_seed)
  X, y, g = [], [], []
  for rep in range(cfg.vf_reps):
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    fs, pr = [], []
    for _ in range(tc.T):
      fs.append(phi(env.o))
      pr.append(env.step(pol.act(env.o))["profit"])
    for t in range(tc.T - tc.vf_H + 1):
      X.append(fs[t])
      y.append(sum(pr[t:t + tc.vf_H]))
      g.append(rep)
  return np.array(X), np.array(y), np.array(g)


# 검증 표본 여부: 학습 반복 번호가 뒤 vf_val 비율에 드는 행
def val_mask(cfg, g):
  return g >= round(cfg.vf_reps * (1 - cfg.vf_val))
