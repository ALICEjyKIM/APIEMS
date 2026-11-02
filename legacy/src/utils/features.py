"""시장 요약 지표.
주문자·공급자 정보를 합계·평균·개수로 요약한 2 + 5 × 품목 수 + 2개 값이며, 선형과 MLP가 같은 입력으로 쓴다.
참여자 하나를 뺀 관측(drop)의 지표로 유지 가치 c_i = V(s) − V(s에서 i 제외)를 계산한다.
"""
from dataclasses import replace
import numpy as np


# 품목별 평균: 그 품목을 가진 참여자만 평균 (없으면 0)
def _mean(val, has):
  n = has.sum(0)
  return np.where(n > 0, (val * has).sum(0) / np.maximum(n, 1), 0.0)


# 관측 o의 시장 요약 지표: 주문자 수, 공급자 수, 품목별 총 요구량·총 공급가능량·평균 제안가격·평균 공급가격·대체 공급자 수,
# 주문자·공급자 평균 재참여확률 (참여자가 없으면 0)
def phi(o):
  hb, hs = o.q > 0, o.cap > 0
  return np.r_[len(o.bid), len(o.sid), o.q.sum(0), o.cap.sum(0), _mean(o.p, hb), _mean(o.c, hs), hs.sum(0),
               o.rb.sum() / max(len(o.rb), 1), o.rs.sum() / max(len(o.rs), 1)].astype(float)


# 관측 o에서 종류 kind의 k번째 참여자를 뺀 관측
def drop(o, kind, k):
  if kind == "buy":
    keep = np.arange(len(o.bid)) != k
    return replace(o, q=o.q[keep], p=o.p[keep], bid=[i for i, m in zip(o.bid, keep) if m], rb=o.rb[keep])
  keep = np.arange(len(o.sid)) != k
  return replace(o, cap=o.cap[keep], c=o.c[keep], sid=[i for i, m in zip(o.sid, keep) if m], rs=o.rs[keep])


# 참여자를 하나씩 뺀 관측들의 지표 행렬: 주문자 순서대로, 이어서 공급자 순서대로 ((B + J) × 지표 수)
def phi_drops(o):
  rows = [phi(drop(o, "buy", k)) for k in range(len(o.bid))] + [phi(drop(o, "sup", k)) for k in range(len(o.sid))]
  return np.array(rows).reshape(-1, len(phi(o)))
