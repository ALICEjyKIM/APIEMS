"""여지 점검: 상태마다 후보 12개를 미래 8번(서로 다른 도착·변동·재참여 난수, 후보끼리는 같은 난수)으로 끝까지 돌려 평균을 낸다.
여지 = 상태별 평균 이득 최대 후보의 고정 padding 대비 차이. 미래 앞 4번으로 고르고 뒤 4번으로 잰 값(교차)도 함께 적어 운으로 부풀린 몫을 뺀다.
상태는 반복 4000~4019의 행동 정책 경로에서 t ∈ {2, 8, 14, 20, 26}. 결과 result/runs/check_headroom.json. 실행: src/에서 python -m result.headroom
"""
from copy import deepcopy
import json
from multiprocessing import Pool
import numpy as np
from utils.params import Cfg
from env.sim import Sim, cands
from result.collect import behave

REPS, TS, K = range(4000, 4020), (2, 8, 14, 20, 26), 8


# 미래 k의 난수 번호 (다른 반복과 겹치지 않게)
def fut(rep, k):
  return 10 ** 6 + rep * K + k


# 상태 하나: 미래 k마다 후보별 남은 누적 이윤
def one(args):
  cfg, rep, t = args
  C, s, pol = cands(cfg), Sim(cfg, rep), behave(cfg, rep)
  while s.t < t:
    s.step(*pol(s.pre()))
  R = np.zeros((K, len(C)))
  for k in range(K):
    for i, a in enumerate(C):
      b = deepcopy(s)
      b.rep, bp = fut(rep, k), behave(cfg, fut(rep, k))
      r = b.step(*a)["Pi"]
      while b.t < cfg.T:
        r += b.step(*bp(b.pre()))["Pi"]
      R[k, i] = r
  return dict(rep=rep, t=t, R=R.tolist())


def main():
  cfg = Cfg()
  with Pool(cfg.n_proc) as p:
    rows = p.map(one, [(cfg, r, t) for r in REPS for t in TS], chunksize=1)
  A = np.array([np.array(x["R"]) - np.array(x["R"])[:, :1] for x in rows])
  m = A.mean(1)
  i = A[:, :K // 2].mean(1).argmax(1)
  cross = A[:, K // 2:].mean(1)[np.arange(len(A)), i]
  one_ = A[:, 0].max(1)
  se = A.std(1, ddof=1) / np.sqrt(K)
  out = dict(cfg=repr(cfg), n=len(rows), K=K, head8=float(m.max(1).mean()), head_cross=float(cross.mean()), head_cross_ci=float(1.96 * cross.std(ddof=1) / np.sqrt(len(cross))),
             head1=float(one_.mean()), mean_by_cand=m.mean(0).tolist(), best_is_fixed=float((m.argmax(1) == 0).mean()), se_mean=float(se.mean()),
             by_t={str(t): float(cross[[x["t"] == t for x in rows]].mean()) for t in TS}, rows=rows)
  json.dump(out, open("result/runs/check_headroom.json", "w"), indent=1)
  print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "cfg")}, indent=1))


if __name__ == "__main__":
  main()
