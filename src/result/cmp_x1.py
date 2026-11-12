"""×1 이후 손실 추정 방법 비교 (ref/decisions.md 2026-11-13 "×1 이후 손실 추정 방법 비교"): MC 평균표 · MLP · GNN을 같은 상태 · 같은 후보(θ = 0)로 비교한다.
pred: adv 원재료 상태의 예측 지표(R², RMSE, 순위 상관, 최선 적중, regret, 고른 후보의 실제 A), ev: 평가 반복 0~29의 30기간 누적 이윤(같은 공통 난수).
결과 result/runs/cmp_x1_pred.json, cmp_x1_ev.json. 실행: src/에서 train_q3 다음 python -m result.cmp_x1 pred, python -m result.cmp_x1 ev
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
import numpy as np
import torch
from utils.params import Cfg
from env.sim import run, cands
from policy.base import fixed, table
from policy.qpol import qpol
from result.main1 import rec
from result.train_q3 import predict
from result.exp_long_x2 import BINS, lbar
from result.check_dev_ci import cci
from result.check_state import jack
from value.raw import load_adv

DEV1 = "result/runs/check_dev.json"
KIND = dict(lin="tab", mlp="tab", mlps="struct", gnn="graph")


# θ = 0 후보만 (Q̂ 정책의 only)
def th0(m, th):
  return th == 0


# θ = 0 후보의 열 번호와 m
def cols(cfg):
  C = cands(cfg)
  i = [k for k, (m, th) in enumerate(C) if th == 0]
  return i, [C[k][0] for k in i]


# MC 평균표 예측: 평균 추정 입찰가의 예상 몫 차이 + 기간 구간의 평균 이후 손실 (m = 1 대비)
def mc(cfg, A, Lb, i):
  G = A["X"][:, i, 20]
  b = np.maximum(np.searchsorted(BINS, A["t"] - (cfg.T - cfg.T_lbar), side="right") - 1, 0)
  return G - G[:, :1] + np.asarray(Lb)[b]


# 상태별 지표: 고른 후보의 실제 A, regret, 최선 적중, 순위 상관(값이 하나뿐이면 nan), m > 1 열의 제곱오차 합
def per(v, Y):
  a, i = np.arange(len(Y)), v.argmax(1)
  pick, best = Y[a, i], Y.max(1)
  rv, ry = (x.argsort(1).argsort(1).astype(float) for x in (v, Y))
  rv, ry = rv - rv.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
  ok = (v.std(1) > 0) & (Y.std(1) > 0)
  rho = np.where(ok, (rv * ry).sum(1) / np.sqrt((rv ** 2).sum(1) * (ry ** 2).sum(1)), np.nan)
  return dict(pick=pick, regret=best - pick, hit=(pick >= best - 1e-9).astype(float), rho=rho, se=((v - Y)[:, 1:] ** 2).sum(1))


# 방법 하나의 지표와 MC 대비 상태별 차이 (반복 군집 95%)
def summ(v, Y, g, base=None):
  p = per(v, Y)
  r2 = lambda k: 1 - p["se"][k].sum() / ((Y[k, 1:] - Y[k, 1:].mean()) ** 2).sum()
  ok = ~np.isnan(p["rho"])
  out = dict(r2=jack(r2, g), rmse=float(np.sqrt(p["se"].sum() / Y[:, 1:].size)), rho=cci(p["rho"][ok], g[ok]),
             **{k: cci(p[k], g) for k in ("pick", "regret", "hit")}, pick_m1=float((v.argmax(1) == 0).mean()))
  if base is not None:
    q = per(base, Y)
    both = ok & ~np.isnan(q["rho"])
    out["vs_mc"] = dict(rho=cci(p["rho"][both] - q["rho"][both], g[both]), **{k: cci(p[k] - q[k], g) for k in ("pick", "regret", "hit", "se")})
  return out


# 예측 비교: adv 상태의 θ = 0 후보 4개, Q̂는 m = 1 열을 빼서 차이로 맞춘다
def pred(cfg):
  i, ms = cols(cfg)
  A, Lb = load_adv(cfg), lbar(ms, (DEV1,))
  assert (A["X"][:, i, 18] == 0).all() and np.allclose(A["X"][0, i, 17], ms) and all(r[0] == 0 for r in Lb)
  Y, g = A["Y"][:, i], A["rep"]
  assert (Y[:, 0] == 0).all()
  V = {"mc": mc(cfg, A, Lb, i)}
  Qs, tr = pickle.load(open("result/data/qhat_q3.pkl", "rb")), json.load(open("result/runs/train_q3.json"))
  for nm in KIND:
    v = predict(Qs[nm], nm, A)[:, i]
    V[nm] = v - v[:, :1]
  res = {nm: summ(v, Y, g, None if nm == "mc" else V["mc"]) for nm, v in V.items()}
  for nm in KIND:
    res[nm]["train_sec"], res[nm]["setting"] = tr["chosen"][nm]["sec"], {k: tr["chosen"][nm][k] for k in ("lam", "L", "H", "lr") if k in tr["chosen"][nm]}
  out = dict(cfg=repr(cfg), n_states=int(len(Y)), n_reps=int(len(np.unique(g))), Lbar=Lb, y_sd=float(Y[:, 1:].std()), res=res)
  json.dump(out, open("result/runs/cmp_x1_pred.json", "w"), indent=1)
  print(json.dumps({k: out[k] for k in ("n_states", "n_reps", "Lbar", "y_sd")} | {"res": res}, indent=1))


# 반복 하나를 정책 하나로 돌린 기록 요약과 걸린 시간
def one(args):
  cfg, rep, name, q, Lb = args
  torch.set_num_threads(1)
  pol = fixed(*cfg.best_fixed) if name == "fixed" else table(cfg, Lb, BINS, "mean") if name == "mc" else qpol(cfg, q, only=th0, kind=KIND[name])
  t0 = time.time()
  return dict(**rec(name, rep, run(cfg, rep, pol)), sec=time.time() - t0)


# 누적 이윤 비교: 고정 padding · MC 평균표 · Q̂ MLP · Q̂ GNN, 고정 padding은 exp_main2와 반복마다 같아야 한다
def ev(cfg):
  _, ms = cols(cfg)
  Lb, Qs = lbar(ms, (DEV1,)), pickle.load(open("result/data/qhat_q3.pkl", "rb"))
  pols = [("fixed", None), ("mc", None), ("mlp", Qs["mlp"]), ("gnn", Qs["gnn"])]
  with Pool(cfg.n_proc) as p:
    rows = p.map(one, [(cfg, r, n, q, Lb) for n, q in pols for r in range(cfg.reps)], chunksize=1)
  tot = {n: np.array([sum(x["Pi"]) for x in rows if x["name"] == n]) for n, _ in pols}
  ref = {x["rep"]: sum(x["Pi"]) for x in json.load(open("result/runs/exp_main2.json"))["rows"] if x["name"] == "fixed"}
  assert np.allclose(tot["fixed"], [ref[r] for r in range(cfg.reps)])  # 같은 시장 · 같은 난수
  g, summary = np.arange(cfg.reps), {}
  for n in tot:
    R = [x for x in rows if x["name"] == n]
    ms_ = np.concatenate([x["m"] for x in R])
    summary[n] = dict(Pi=float(tot[n].mean()), vs_fixed=cci(tot[n] - tot["fixed"], g), vs_mc=cci(tot[n] - tot["mc"], g),
                      win_fixed=int((tot[n] > tot["fixed"]).sum()), win_mc=int((tot[n] > tot["mc"]).sum()),
                      m_gt1=float((ms_ > 1).mean()), sec_period=float(np.mean([x["sec"] for x in R]) / cfg.T), sw=float(np.mean([sum(x["sw"]) for x in R])))
  json.dump(dict(cfg=repr(cfg), Lbar=Lb, summary=summary, rows=rows), open("result/runs/cmp_x1_ev.json", "w"), indent=1)
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  {"pred": pred, "ev": ev}[sys.argv[1]](Cfg())
