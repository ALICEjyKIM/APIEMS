"""Q̂ 정책: 후보마다 Q̂ 입력을 만들어 점수가 가장 큰 후보를 고른다 (동점이면 cands 순서상 앞, 즉 작은 m·θ).
입력 형태 kind는 표(tab: 선형·MLP), 표 + 구조 요약(struct: MLP + 구조 요약), (표, 그래프 목록)(graph: GNN)이고 예상 청산은 한 번만 푼다.
only로 후보를 제한하면 2 × 2의 padding만(θ = 0), 쿠폰만(m = 1) 정책이 된다. 가격 기록이 없는 기간 0은 한 기간 최적과 같이 고정 padding이다.
"""
from env.sim import cands
from value.feat import feats, plan, ready
from value.graph import feats_s, graph


# 방법별 Q̂ 입력
def build(cfg, o, kind):
  P = plan(cfg, o)
  if kind == "struct":
    return feats_s(cfg, o, P)
  X = feats(cfg, o, P)
  return (X, graph(cfg, o, P)) if kind == "graph" else X


# 점수 함수 f(입력) → 후보별 Â로 고르는 정책
def qpol(cfg, f, only=None, kind="tab"):
  C = cands(cfg)
  ok = [i for i, (m, th) in enumerate(C) if only is None or only(m, th)]
  def pol(o):
    if not ready(o):
      return 1.0, 0.0
    v = f(build(cfg, o, kind))
    return C[max(ok, key=lambda i: v[i])]
  return pol
