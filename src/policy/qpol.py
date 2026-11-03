"""Q̂ 정책: 후보마다 Q̂ 입력(value.feat)을 만들어 점수가 가장 큰 후보를 고른다 (동점이면 cands 순서상 앞, 즉 작은 m·θ).
only로 후보를 제한하면 2 × 2의 padding만(θ = 0), 쿠폰만(m = 1) 정책이 된다.
가격 기록이 없는 기간 0은 한 기간 최적과 같이 고정 padding이다.
"""
import numpy as np
from env.sim import cands
from value.feat import feats, ready


# 점수 함수 f(입력 행렬) → 후보별 Â로 고르는 정책
def qpol(cfg, f, only=None):
  C = cands(cfg)
  ok = [i for i, (m, th) in enumerate(C) if only is None or only(m, th)]
  def pol(o):
    if not ready(o):
      return 1.0, 0.0
    v = f(feats(cfg, o))
    return C[max(ok, key=lambda i: v[i])]
  return pol
