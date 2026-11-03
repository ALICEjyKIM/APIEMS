"""최선 고정값 튜닝 (N4): 후보 (m, θ)마다 30기간 고정으로 튜닝 반복을 돌려 누적 이윤 평균이 가장 큰 후보를 고른다.
튜닝 반복 1000~1029는 평가 반복(0~29)과 분리하고, 공통 난수로 같은 반복끼리 1위와의 차이를 잰다 (ref/decisions.md "최선 고정값 튜닝").
결과는 result/runs/tune_fixed.json. 실행: src/에서 python -m result.tune_fixed
"""
import json
from multiprocessing import Pool
import numpy as np
from utils.params import Cfg
from env.sim import run, cands
from policy.base import fixed

REPS = range(1000, 1030)


# 반복 하나를 고정 후보로 돌린 기간별 기록 요약
def one(args):
  cfg, rep, m, th = args
  R = run(cfg, rep, fixed(m, th))
  a = lambda k, f=float: [f(r[k]) for r in R]
  return dict(m=m, th=th, rep=rep, Pi=a("Pi"), G=a("G"), cpn=a("cpn"), paid=a("paid"), sw=a("sw"), win=a("win", int), nb=a("nb", int), ns=a("ns", int))


def main():
  cfg = Cfg()
  with Pool(10) as p:
    rows = p.map(one, [(cfg, r, m, th) for m, th in cands(cfg) for r in REPS], chunksize=1)
  tot = {k: np.array([sum(x["Pi"]) for x in rows if (x["m"], x["th"]) == k]) for k in cands(cfg)}
  best = max(tot, key=lambda k: tot[k].mean())
  summ = []
  for (m, th), v in tot.items():
    R = [x for x in rows if (x["m"], x["th"]) == (m, th)]
    d = v - tot[best]
    summ.append(dict(m=m, th=th, Pi=float(v.mean()), sd=float(v.std(ddof=1)), diff=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                     win=float(np.mean([sum(x["win"]) / sum(x["nb"]) for x in R])), nb=float(np.mean([np.mean(x["nb"]) for x in R])),
                     ns=float(np.mean([np.mean(x["ns"]) for x in R])), cpn=float(np.mean([sum(x["cpn"]) for x in R])), sw=float(np.mean([sum(x["sw"]) for x in R]))))
  out = dict(cfg=repr(cfg), reps=list(REPS), best=list(best), summary=summ, rows=rows)
  json.dump(out, open("result/runs/tune_fixed.json", "w"), indent=1)
  for s in sorted(summ, key=lambda s: -s["Pi"]):
    print(s)


if __name__ == "__main__":
  main()
