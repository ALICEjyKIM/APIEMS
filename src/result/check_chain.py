"""제외 연쇄 점검 (ref/decisions.md "제외 연쇄 점검"): 이번 기간만 m(θ = 0)으로 청산하고 이후 고정 padding으로 끝까지 돌리며 padding 제외 인원 → 다음 기간 참여자 수 → 이후 누적 참여 · 몫을 기록한다.
"제외된 참여자는 덜 돌아온다"는 재참여 모형의 가정이고, 이 점검은 그 뒤의 연결(참여 감소가 플랫폼 몫을 얼마나 줄이나)을 잰다. 예상 청산의 제외 인원(입찰 전에 아는 값)도 함께 둔다.
결과 result/runs/check_chain_{규모}_{태그}.json. 실행: src/에서 python -m result.check_chain x1 [키=값 ...] (예: ex_p=0.32, 없으면 태그 base)
"""
from copy import deepcopy
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import Sim
from mech.chu import clear
from policy.base import est
from result.check_pred import REPS, T0
from result.headroom import fut, K
from result.check_dev_ci import cci

KEYS = ("I", "exb", "exs", "nb1", "ns1", "pb", "ps", "win", "R")


# 상태 하나: 후보 m별 예상 제외 인원, 미래 k × 후보 m마다 이번 기간 몫 · 실제 제외 인원 · 다음 기간 참여자 수 · 이후 누적 참여 · 성립 주문 · 남은 누적 순이윤
def one(args):
  cfg, rep, t = args
  s = Sim(cfg, rep)
  while s.t < t:
    s.step(1.0, 0.0)
  o, ms = s.pre(), list(cfg.m_grid)
  We, ge = est(o)
  pu = [[int(p.Ub.sum()), int(p.Uj.sum())] for p in (clear(cfg, o.q, We, o.K, ge, m) for m in ms)]
  out = {k: np.zeros((K, len(ms))) for k in KEYS}
  for k in range(K):
    for i, m in enumerate(ms):
      b = deepcopy(s)
      b.rep = fut(rep, k)
      rec = b.step(m, 0.0)
      v = dict(I=rec["G"], exb=rec["Ub"].sum(), exs=rec["Uj"].sum(), nb1=len(b.bid), ns1=len(b.slot), pb=0, ps=0, win=0, R=rec["Pi"])
      while b.t < cfg.T:
        r = b.step(1.0, 0.0)
        v["pb"], v["ps"], v["win"], v["R"] = v["pb"] + r["nb"], v["ps"] + r["ns"], v["win"] + r["win"], v["R"] + r["Pi"]
      for key in KEYS:
        out[key][k, i] = v[key]
  return dict(rep=rep, t=t, pu=pu, **{k: v.tolist() for k, v in out.items()})


# 최소제곱 기울기와 설명력: y ~ 1 + X
def ols(X, y):
  A = np.c_[np.ones(len(y)), X]
  b = np.linalg.lstsq(A, y, rcond=None)[0]
  return dict(coef=b[1:].tolist(), icpt=float(b[0]), r2=float(1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


# 요약: m = 1 대비 차이(미래 평균)로 연결의 각 단계와 회귀, 이번 기간 이득이 양수인 상태의 끝까지 차이
def summ(rows, T):
  g = np.array([r["rep"] for r in rows])
  d = {k: np.array([np.array(r[k]).mean(0) for r in rows]) for k in KEYS}
  d = {k: v - v[:, :1] for k, v in d.items()}
  d["D"], d["pu_b"], d["pu_s"] = d["R"], *(np.array([[p[j] for p in r["pu"]] for r in rows], float) for j in (0, 1))
  d["pu_b"], d["pu_s"] = d["pu_b"] - d["pu_b"][:, :1], d["pu_s"] - d["pu_s"][:, :1]
  d["L"] = d["D"] - d["I"]
  rem = np.array([T - r["t"] - 1 for r in rows], float)
  by_m = {k: [cci(d[k][:, i], g) for i in range(1, d[k].shape[1])] for k in ("I", "L", "D", "exb", "exs", "nb1", "ns1", "pb", "ps", "win", "pu_b", "pu_s")}
  M = d["I"].shape[1]
  flat = lambda k: np.concatenate([d[k][:, i] for i in range(1, M)])
  remf = np.tile(rem, M - 1)
  reg = dict(next_b=ols(flat("exb")[:, None], flat("nb1")), next_s=ols(flat("exs")[:, None], flat("ns1")),
             part=ols(np.c_[flat("pb"), flat("ps")], flat("L")), ex=ols(np.c_[flat("exb"), flat("exs")], flat("L")),
             ex_rem=ols(np.c_[flat("exb") * remf, flat("exs") * remf], flat("L")), pred=ols(np.c_[flat("pu_b") * remf, flat("pu_s") * remf], flat("L")))
  pos = []
  for i in range(1, M):
    k = d["I"][:, i] > 0
    pos.append(dict(share=float(k.mean()), I=cci(d["I"][k, i], g[k]), L=cci(d["L"][k, i], g[k]), D=cci(d["D"][k, i], g[k]),
                    ratio=float(-d["L"][k, i].mean() / d["I"][k, i].mean())))
  return dict(n=len(rows), by_m=by_m, reg=reg, pos=pos)


def main():
  sc, ov = sys.argv[1], dict(a.split("=") for a in sys.argv[2:])
  cfg = replace(Cfg(), **SCALE[sc], **{k: float(v) for k, v in ov.items()})
  tag = "_".join(f"{k}{v}" for k, v in ov.items()) or "base"
  rows = []
  with Pool(cfg.n_proc) as p:
    for x in p.imap_unordered(one, [(cfg, r, t) for r in REPS for t in range(T0, cfg.T)]):
      rows.append(x)
      print(len(rows), x["rep"], x["t"], flush=True)
  rows.sort(key=lambda x: (x["rep"], x["t"]))
  out = summ(rows, cfg.T)
  json.dump(dict(cfg=repr(cfg), scale=sc, tag=tag, K=K, **out, rows=rows), open(f"result/runs/check_chain_{sc}_{tag}.json", "w"), indent=1)
  print(json.dumps(dict(reg=out["reg"], pos=out["pos"]), indent=1))


if __name__ == "__main__":
  main()
