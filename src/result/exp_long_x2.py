"""×2 장기 표 규칙 실험 (ref/decisions.md "×2 장기 표 규칙 실험"): 평가 반복 0~29에서 고정 padding, 한 기간 최적(평균 · 시나리오), 장기 표 규칙(평균 · 시나리오)을 같은 공통 난수로 30기간 돌린다.
이후 손실 표 Lbar는 한 번 올린 대가 점검 ×2(check_dev_x2.json, 반복 1000~1009)에서 시점 구간별 · m별 평균 L = D − I로 만든다. 평가 반복과 겹치지 않는다.
결과 result/runs/exp_long_{규모}[_정책들].json. 실행: src/에서 python -m result.exp_long_x2 [규모] [정책 이름 ...] (규모 기본 x2, 정책 없으면 다섯 개 모두. 표 규칙은 x2에서만)
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import run
from policy.base import fixed, myopic, saa, table

BINS = (0, 10, 15, 20, 25)
NAMES = ("table_saa", "saa", "table_mean", "myopic", "fixed")


# 이후 손실 표: 시점 구간 [BINS[k], BINS[k+1])마다 m별 평균 (D − I), m = 1은 0 (기간 0~4는 첫 구간의 자료 t = 5~9로 정해진다)
def lbar(path):
  rows, out = json.load(open(path))["rows"], []
  for k, lo in enumerate(BINS):
    R = [r for r in rows if lo <= r["t"] < (BINS[k + 1] if k + 1 < len(BINS) else 10 ** 9)]
    I, D = np.array([r["I"] for r in R]), np.array([np.mean(r["R"], 0) for r in R])
    out.append(((D - D[:, :1]) - (I - I[:, :1])).mean(0).tolist())
  return out


# 정책 이름 → 정책 (작업자 안에서 만든다)
def pol(cfg, name, L):
  return dict(fixed=lambda: fixed(1.0, 0.0), myopic=lambda: myopic(cfg), saa=lambda: saa(cfg),
              table_mean=lambda: table(cfg, L, BINS, "mean"), table_saa=lambda: table(cfg, L, BINS, "saa"))[name]()


# 반복 하나를 정책 하나로 30기간 돌린 기록 요약
def one(args):
  cfg, rep, name, L = args
  R = run(cfg, rep, pol(cfg, name, L))
  a = lambda k: [float(r[k]) for r in R]
  return dict(name=name, rep=rep, Pi=a("Pi"), G=a("G"), paid=a("paid"), sw=a("sw"), m=a("m"), nb=a("nb"), ns=a("ns"), win=a("win"))


# 짝지은 차이의 평균과 95% 구간
def pd(x, y):
  d = x - y
  return dict(mean=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))))


def main():
  sc = sys.argv[1] if sys.argv[1:2] and sys.argv[1] in SCALE else "x2"
  cfg, names = replace(Cfg(), **SCALE[sc]), tuple(a for a in sys.argv[1:] if a not in SCALE) or NAMES
  tab = any(n.startswith("table") for n in names)
  assert sc == "x2" or not tab  # 이후 손실 표는 ×2 점검에서 만든 값
  L, rows = lbar("result/runs/check_dev_x2.json") if tab else None, []
  with Pool(cfg.n_proc) as p:
    for x in p.imap_unordered(one, [(cfg, r, n, L) for n in names for r in range(cfg.reps)]):
      rows.append(x)
      print(len(rows), x["name"], x["rep"], flush=True)
  rows.sort(key=lambda x: (names.index(x["name"]), x["rep"]))
  P = {n: np.array([x["Pi"] for x in rows if x["name"] == n]) for n in names}
  M = {n: np.array([x["m"] for x in rows if x["name"] == n]) for n in names}
  e, summ = BINS[-1], {}
  for n in names:
    summ[n] = dict(Pi=float(P[n].sum(1).mean()), m_gt1=float((M[n] > 1).mean()), m_mean=float(M[n].mean()))
    for b in ("fixed", "myopic", "saa"):
      if b in P:
        summ[n]["vs_" + b] = pd(P[n].sum(1), P[b].sum(1))
    if "fixed" in P:
      summ[n]["early_vs_fixed"], summ[n]["late_vs_fixed"] = pd(P[n][:, :e].sum(1), P["fixed"][:, :e].sum(1)), pd(P[n][:, e:].sum(1), P["fixed"][:, e:].sum(1))
  tag = "" if names == NAMES else "_" + "_".join(names)
  json.dump(dict(cfg=repr(cfg), Lbar=L, bins=BINS, summary=summ, rows=rows), open(f"result/runs/exp_long_{sc}{tag}.json", "w"), indent=1)
  print(json.dumps(dict(Lbar=L, summary=summ), indent=1))


if __name__ == "__main__":
  main()
