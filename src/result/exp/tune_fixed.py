"""최선 고정값 튜닝 (N4): 후보 (m, θ)마다 30기간 고정으로 튜닝 반복을 돌려 누적 이윤 평균이 가장 큰 후보를 고른다.
튜닝 반복 1000~1029는 평가 반복(0~29)과 분리하고, 공통 난수로 같은 반복끼리 1위와의 차이를 잰다 (ref/decisions.md "최선 고정값 튜닝").
결과는 result/runs/tune_fixed[_x2][_바꾼Cfg키값].json. 실행: src/에서 python -m result.exp.tune_fixed [x2|x4] [Cfg 키=값 ...] [base=파일] (시장 규모 기본 ×1, base는 이미 돌린 후보를 합쳐 함께 비교)
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import run, cands
from policy.base import fixed
from result.common import ovr

REPS = range(1000, 1030)


# 반복 하나를 고정 후보로 돌린 기간별 기록 요약
def one(args):
  cfg, rep, m, th = args
  R = run(cfg, rep, fixed(m, th))
  a = lambda k, f=float: [f(r[k]) for r in R]
  return dict(m=m, th=th, rep=rep, Pi=a("Pi"), G=a("G"), cpn=a("cpn"), paid=a("paid"), sw=a("sw"), win=a("win", int), nb=a("nb", int), ns=a("ns", int))


def main():
  sc = sys.argv[1] if sys.argv[1:2] and sys.argv[1] in SCALE else "x1"
  ov = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
  base = ov.pop("base", None)
  cfg, rows = ovr(replace(Cfg(), **SCALE[sc]), ov), []
  with Pool(cfg.n_proc) as p:
    for x in p.imap_unordered(one, [(cfg, r, m, th) for m, th in cands(cfg) for r in REPS]):
      rows.append(x)
      print(len(rows), x["m"], x["th"], x["rep"], flush=True)
  # 시뮬레이터는 m_grid를 읽지 않아 다른 m_grid로 돌린 고정 후보 기록도 같은 반복끼리 비교된다
  rows += [x for x in json.load(open(base))["rows"] if (x["m"], x["th"]) not in cands(cfg)] if base else []
  rows.sort(key=lambda x: (x["m"], x["th"], x["rep"]))
  C = sorted({(x["m"], x["th"]) for x in rows})
  assert all([x["rep"] for x in rows if (x["m"], x["th"]) == k] == list(REPS) for k in C)
  tot = {k: np.array([sum(x["Pi"]) for x in rows if (x["m"], x["th"]) == k]) for k in C}
  best = max(tot, key=lambda k: tot[k].mean())
  summ = []
  for (m, th), v in tot.items():
    R = [x for x in rows if (x["m"], x["th"]) == (m, th)]
    d = v - tot[best]
    summ.append(dict(m=m, th=th, Pi=float(v.mean()), sd=float(v.std(ddof=1)), diff=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                     win=float(np.mean([sum(x["win"]) / sum(x["nb"]) for x in R])), nb=float(np.mean([np.mean(x["nb"]) for x in R])),
                     ns=float(np.mean([np.mean(x["ns"]) for x in R])), cpn=float(np.mean([sum(x["cpn"]) for x in R])), sw=float(np.mean([sum(x["sw"]) for x in R]))))
  out = dict(cfg=repr(cfg), scale=sc, base=base, reps=list(REPS), best=list(best), summary=summ, rows=rows)
  tag = ("" if sc == "x1" else "_" + sc) + "".join(f"_{k}{v}" for k, v in ov.items())
  json.dump(out, open(f"result/runs/tune_fixed{tag}.json", "w"), indent=1)
  for s in sorted(summ, key=lambda s: -s["Pi"]):
    print(s)


if __name__ == "__main__":
  main()
