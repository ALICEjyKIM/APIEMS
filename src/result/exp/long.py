"""기간별 평균 이후 손실 반영 정책 실험 (ref/decisions.md "×2 장기 표 규칙 실험"): 평가 반복 0~29에서 고정 padding, 최선 고정값(best), 한 기간 최적(평균 · 시나리오), 기간별 평균 이후 손실 반영 정책(평균 · 시나리오)을 같은 공통 난수로 30기간 돌린다.
이후 손실 표 Lbar는 한 번 올린 대가 점검 ×2(check_dev_x2.json, 반복 1000~1009)에서 기간 구간별 · m별 평균 L = D − I로 만든다(result.common.lbar). 그 파일에 없는 m은 dev=파일(같은 상태 · 같은 미래에서 잰 점검)에서 더한다.
결과 result/runs/exp_long_{규모}[_정책들][_바꾼Cfg키값][_r첫반복-끝반복].json. 실행: src/에서 python -m result.exp.long [규모] [정책 이름 ...] [Cfg 키=값 ...] [dev=파일] (규모 기본 x2, 정책 없으면 best를 뺀 다섯 개, table 정책은 x2에서만, 튜플은 쉼표로. 예: rep0=30 reps=60 m_grid=1,1.05,1.1)
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import os
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import run, cands
from policy.base import fixed, myopic, saa, table
from result.common import BINS, DEV, ovr, lbar, save, part

NAMES = ("table_saa", "saa", "table_mean", "myopic", "fixed")


# 정책 이름 → 정책 (작업자 안에서 만든다)
def pol(cfg, name, L):
  return dict(fixed=lambda: fixed(1.0, 0.0), best=lambda: fixed(*cfg.best_fixed), myopic=lambda: myopic(cfg), saa=lambda: saa(cfg),
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
  ov = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
  dev = ov.pop("dev", None)
  cfg = ovr(replace(Cfg(), **SCALE[sc]), ov)
  names = tuple(a for a in sys.argv[1:] if a not in SCALE and "=" not in a) or NAMES
  tab = any(n.startswith("table") for n in names)
  assert sc == "x2" or not tab  # 이후 손실 표는 ×2 점검에서 만든 값
  L = lbar(sorted({m for m, _ in cands(cfg)}), (DEV,) + ((dev,) if dev else ())) if tab else None
  tag = ("" if names == NAMES else "_" + "_".join(names)) + "".join(f"_{k}{v}" for k, v in ov.items() if k not in ("rep0", "reps", "n_proc")) + (f"_r{cfg.rep0}-{cfg.rep0 + cfg.reps - 1}" if cfg.rep0 else "")
  out = f"result/runs/exp_long_{sc}{tag}"
  # 시장 순서로 넣고 실행 하나가 끝날 때마다 중간 저장: 끝까지 못 가도 끝난 시장은 모든 정책이 함께 남는다.
  # 같은 명령으로 다시 실행하면 저장된 실행은 건너뛴다 (난수가 (반복, 정책)으로 정해져 결과가 같다)
  rows = part(out + ".part.json", cfg)
  done = {(x["name"], x["rep"]) for x in rows}
  with Pool(cfg.n_proc) as p:
    for x in p.imap_unordered(one, [(cfg, r, n, L) for r in range(cfg.rep0, cfg.rep0 + cfg.reps) for n in names if (n, r) not in done]):
      rows.append(x)
      print(len(rows), x["name"], x["rep"], flush=True)
      save(out + ".part.json", dict(cfg=repr(cfg), dev=dev, Lbar=L, bins=BINS, rows=rows))
  rows.sort(key=lambda x: (names.index(x["name"]), x["rep"]))
  P = {n: np.array([x["Pi"] for x in rows if x["name"] == n]) for n in names}
  M = {n: np.array([x["m"] for x in rows if x["name"] == n]) for n in names}
  e, summ = BINS[-1], {}
  for n in names:
    summ[n] = dict(Pi=float(P[n].sum(1).mean()), m_gt1=float((M[n] > 1).mean()), m_mean=float(M[n].mean()))
    for b in ("fixed", "best", "myopic", "saa"):
      if b in P:
        summ[n]["vs_" + b] = pd(P[n].sum(1), P[b].sum(1))
    if "fixed" in P:
      summ[n]["early_vs_fixed"], summ[n]["late_vs_fixed"] = pd(P[n][:, :e].sum(1), P["fixed"][:, :e].sum(1)), pd(P[n][:, e:].sum(1), P["fixed"][:, e:].sum(1))
  json.dump(dict(cfg=repr(cfg), dev=dev, Lbar=L, bins=BINS, summary=summ, rows=rows), open(out + ".json", "w"), indent=1)
  os.remove(out + ".part.json")
  print(json.dumps(dict(Lbar=L, summary=summ), indent=1))


if __name__ == "__main__":
  main()
