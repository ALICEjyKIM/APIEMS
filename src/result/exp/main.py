"""본 실험 (메인·질문 1): 평가 반복 0~29에서 고정 padding(= 최선 고정값), 한 기간 최적, Q̂ 선형, Q̂ MLP, 제안 방법의 padding만·재참여 보상만을 같은 공통 난수로 돌린다.
1차는 Q̂를 train_qhat.json·qhat.pkl에서, 2차(td)는 train_td.json·qhat_td.pkl에서 읽고(result.train.fit_main), 제안 방법은 검증 점수가 높은 근사 방법이다 (ref/decisions.md "본 실험 1차", "TD 학습 (2차)").
결과는 result/runs/exp_main1.json, exp_main2.json. 실행: src/에서 python -m result.exp.main [td]
"""
import json
import pickle
import sys
from multiprocessing import Pool
import numpy as np
from utils.params import Cfg
from env.sim import run
from policy.base import fixed, myopic
from policy.qpol import qpol
from result.common import rec

ONLY = {None: None, "pad": lambda m, th: th == 0, "cpn": lambda m, th: m == 1}


# 반복 하나를 정책 하나로 돌린 기간별 기록 요약
def one(args):
  cfg, rep, name, q, only = args
  pol = fixed(*cfg.best_fixed) if name == "fixed" else myopic(cfg) if name == "myopic" else qpol(cfg, q, ONLY[only])
  return rec(name, rep, run(cfg, rep, pol))


def main():
  cfg, td = Cfg(), sys.argv[1:] == ["td"]
  Qs = pickle.load(open("result/data/qhat_td.pkl" if td else "result/data/qhat.pkl", "rb"))
  prop = json.load(open("result/runs/train_td.json" if td else "result/runs/train_qhat.json"))["proposal"]
  pols = [("fixed", None, None), ("myopic", None, None), ("q_lin", Qs["lin"], None), ("q_mlp", Qs["mlp"], None), ("pad_only", Qs[prop], "pad"), ("cpn_only", Qs[prop], "cpn")]
  with Pool(20) as p:
    rows = p.map(one, [(cfg, r, n, q, o) for n, q, o in pols for r in range(cfg.reps)], chunksize=1)
  tot = {n: np.array([sum(x["Pi"]) for x in rows if x["name"] == n]) for n, _, _ in pols}
  ci = lambda d: float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
  summ = {}
  for n in tot:
    R = [x for x in rows if x["name"] == n]
    ms, th = np.concatenate([x["m"] for x in R]), np.concatenate([x["th"] for x in R])
    summ[n] = dict(Pi=float(tot[n].mean()), vs_fixed=float((tot[n] - tot["fixed"]).mean()), ci_fixed=ci(tot[n] - tot["fixed"]),
                   vs_myopic=float((tot[n] - tot["myopic"]).mean()), ci_myopic=ci(tot[n] - tot["myopic"]),
                   m_gt1=float((ms > 1).mean()), m_mean=float(ms.mean()), th_gt0=float((th > 0).mean()),
                   cpn=float(np.mean([sum(x["cpn"]) for x in R])), sw=float(np.mean([sum(x["sw"]) for x in R])),
                   win=float(np.mean([sum(x["win"]) / sum(x["nb"]) for x in R])), nb=float(np.mean([np.mean(x["nb"]) for x in R])), ns=float(np.mean([np.mean(x["ns"]) for x in R])))
  d = {n: tot[n] - tot["fixed"] for n in tot}
  inter = d["q_" + prop] - d["pad_only"] - d["cpn_only"]
  out = dict(cfg=repr(cfg), proposal="q_" + prop, summary=summ, interaction=dict(mean=float(inter.mean()), ci=ci(inter)), rows=rows)
  json.dump(out, open("result/runs/exp_main2.json" if td else "result/runs/exp_main1.json", "w"), indent=1)
  print(json.dumps(dict(proposal=out["proposal"], summary=summ, interaction=out["interaction"]), indent=1))


if __name__ == "__main__":
  main()
