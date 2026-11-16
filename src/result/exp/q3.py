"""본 실험 (질문 3): 평가 반복 0~29에서 고정 padding, 한 기간 최적, Q̂ 선형·MLP·MLP + 구조 요약·GNN(result.train.fit_q3가 고른 모형)을 같은 공통 난수로 돌린다.
주 지표는 MLP 대비 누적 이윤 차이이고 고정 padding·한 기간 최적 대비도 적는다 (ref/decisions.md "질문 3 비교 범위"). 기간당 결정 시간도 기록한다.
결과 result/runs/exp_q3.json. 실행: src/에서 python -m result.exp.q3
"""
import json
import pickle
import time
from multiprocessing import Pool
import numpy as np
import torch
from utils.params import Cfg
from env.sim import run
from policy.base import fixed, myopic
from policy.qpol import qpol
from result.common import rec

KIND = dict(lin="tab", mlp="tab", mlps="struct", gnn="graph")


# 반복 하나를 정책 하나로 돌린 기록 요약과 걸린 시간
def one(args):
  cfg, rep, name, q = args
  torch.set_num_threads(1)
  pol = fixed(*cfg.best_fixed) if name == "fixed" else myopic(cfg) if name == "myopic" else qpol(cfg, q, kind=KIND[name[2:]])
  t0 = time.time()
  return dict(**rec(name, rep, run(cfg, rep, pol)), sec=time.time() - t0)


def main():
  cfg = Cfg()
  Qs = pickle.load(open("result/data/qhat_q3.pkl", "rb"))
  prop = json.load(open("result/runs/train_q3.json"))["proposal"]
  pols = [("fixed", None), ("myopic", None)] + [("q_" + m, Qs[m]) for m in KIND if m in Qs]
  with Pool(cfg.n_proc) as p:
    rows = p.map(one, [(cfg, r, n, q) for n, q in pols for r in range(cfg.reps)], chunksize=1)
  tot = {n: np.array([sum(x["Pi"]) for x in rows if x["name"] == n]) for n, _ in pols}
  ci = lambda d: float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
  summ = {}
  for n in tot:
    R = [x for x in rows if x["name"] == n]
    ms, th = np.concatenate([x["m"] for x in R]), np.concatenate([x["th"] for x in R])
    summ[n] = dict(Pi=float(tot[n].mean()), **{f"vs_{b}": float((tot[n] - tot[b]).mean()) for b in ("fixed", "myopic", "q_mlp") if b in tot},
                   **{f"ci_{b}": ci(tot[n] - tot[b]) for b in ("fixed", "myopic", "q_mlp") if b in tot},
                   m_gt1=float((ms > 1).mean()), m_mean=float(ms.mean()), th_gt0=float((th > 0).mean()),
                   cpn=float(np.mean([sum(x["cpn"]) for x in R])), sw=float(np.mean([sum(x["sw"]) for x in R])),
                   win=float(np.mean([sum(x["win"]) / sum(x["nb"]) for x in R])), nb=float(np.mean([np.mean(x["nb"]) for x in R])),
                   ns=float(np.mean([np.mean(x["ns"]) for x in R])), sec_period=float(np.mean([x["sec"] for x in R]) / cfg.T))
  out = dict(cfg=repr(cfg), proposal="q_" + prop, summary=summ, rows=rows)
  json.dump(out, open("result/runs/exp_q3.json", "w"), indent=1)
  print(json.dumps(dict(proposal=out["proposal"], summary=summ), indent=1))


if __name__ == "__main__":
  main()
