"""가치 근사 학습 데이터.
튜닝 seed(평가 seed와 분리)에서 후속 정책(규칙 기반, Cfg 몫)으로 돌린 rollout의 (시장 요약 지표, 이후 vf_H기간 이윤 합)을 모은다.
검증 분할(학습 반복의 뒤 vf_val 비율)과 설정 선택(select)은 모든 근사 방법이 같은 것을 쓴다.
"""
from dataclasses import replace
import numpy as np
from env.platform import Env
from bench.rule_split import rule
from utils.features import phi


# 튜닝 seed 규칙 기반 rollout vf_reps반복의 (관측 목록, 이후 vf_H기간 이윤 합, 반복 번호). 창이 T 안에 드는 t ≤ T − vf_H만
# 여러 모델이 같은 데이터로 학습하도록 관측을 돌려주고, 입력 변환은 모델마다 한다
def collect_obs(cfg):
  tc = replace(cfg, seed=cfg.tune_seed)
  O, y, g = [], [], []
  for rep in range(cfg.vf_reps):
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    os_, pr = [], []
    for _ in range(tc.T):
      os_.append(env.o)
      pr.append(env.step(pol.act(env.o))["profit"])
    for t in range(tc.T - tc.vf_H + 1):
      O.append(os_[t])
      y.append(sum(pr[t:t + tc.vf_H]))
      g.append(rep)
  return O, np.array(y), np.array(g)


# 시장 요약 지표 학습 데이터 (지표 행렬, 이후 vf_H기간 이윤 합, 반복 번호)
def collect(cfg):
  O, y, g = collect_obs(cfg)
  return np.array([phi(o) for o in O]), y, g


# 검증 표본 여부: 학습 반복 번호가 뒤 vf_val 비율에 드는 행
def val_mask(cfg, g):
  return g >= round(cfg.vf_reps * (1 - cfg.vf_val))


# 설정 선택: grid의 각 설정 a로 make(a)를 학습 분할에 맞춰 검증 평균제곱오차를 재고, 가장 작은 설정으로 전체 데이터를 다시 맞춘다
def select(cfg, make, grid, X, y, g):
  va = val_mask(cfg, g)
  mse = {a: float(np.mean((make(a).fit(X[~va], y[~va]).predict(X[va]) - y[va]) ** 2)) for a in grid}
  a = min(mse, key=mse.get)
  return make(a).fit(X, y), mse
