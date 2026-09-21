"""재참여확률의 구간선형 근사.
잉여율 기준 꺾인 점을 만들고, MILP에서는 참여자별 제안 금액을 곱해 잉여 단위로 쓴다 (addGenConstrPWL).
꺾인 점: 잉여율 0 ~ 확률 pwl_hi 지점 등간격 pwl_n점 + 잉여율 pwl_rmax 끝점.
"""
import numpy as np


# 반응 함수 resp의 종류 kind에 대한 잉여율 꺾인 점과 재참여확률
def points(cfg, resp, kind):
  r = np.r_[np.linspace(0, resp.rate(kind, cfg.pwl_hi), cfg.pwl_n), cfg.pwl_rmax]
  assert r[-2] < r[-1]
  return r, resp.prob(kind, r, 1.0)
