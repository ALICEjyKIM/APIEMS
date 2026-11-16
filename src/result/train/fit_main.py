"""result.exp.main이 쓰는 Q̂ 학습 (N5 1차 · 2차): 행동 정책(고정 padding, 확률 eps로 후보 무작위)의 경로에서 기간 t ≥ 1마다 시장을 복사해 후보 12개로 분기한 자료를 모으고, 선형·MLP를 탐색 예산 8개씩 학습한다.
collect: 분기 뒤 행동 정책으로 끝까지 돌린 고정 padding 대비 남은 누적 이윤 차이 A → result/data/adv.npz. collect_td: 한 기간만 진행한 Π와 다음 상태 입력 → result/data/td/{rep}.npz. train_qhat: adv.npz로 1차 학습. train_td: td 자료로 기간을 거꾸로 2차 학습(expected SARSA).
결과 result/runs/train_qhat.json · train_td.json, 모형 result/data/qhat.pkl · qhat_td.pkl. 실행: src/에서 python -m result.train.fit_main collect|collect_td|train_qhat|train_td (ref/design.md 5장, ref/decisions.md "Q̂ 탐색 예산", "TD 학습 (2차)")
"""
from copy import deepcopy
from glob import glob
from multiprocessing import Pool
import json
import os
import pickle
import sys
import time
import numpy as np
from utils.params import Cfg
from env.sim import Sim, cands
from value.feat import feats, ready
from value.fit import QT, scaler, ridge, mlp, score, target
from result.common import behave

DIR = "result/data/td"
S, A = 17, 11  # 입력의 상태·후보 열 수 (value/feat.py)


# 반복 하나 (collect): 상태마다 입력과 후보별 남은 누적 이윤
def collect_one(args):
  cfg, rep = args
  t0, C = time.time(), cands(cfg)
  assert C[0] == (1.0, 0.0)
  s, pol, X, Y, T, R0 = Sim(cfg, rep), behave(cfg, rep), [], [], [], []
  while s.t < cfg.T:
    o = s.pre()
    if o.t > 0 and ready(o):
      R = []
      for a in C:
        b, r = deepcopy(s), 0.0
        r += b.step(*a)["Pi"]
        while b.t < cfg.T:
          r += b.step(*pol(b.pre()))["Pi"]
        R.append(r)
      X.append(feats(cfg, o))
      Y.append(np.array(R) - R[0])
      T.append(o.t)
      R0.append(R[0])
    s.step(*pol(o))
  print(rep, len(T), round(time.time() - t0), flush=True)
  return np.array(X), np.array(Y), np.full(len(T), rep), np.array(T), np.array(R0)


# 반복 하나 (collect_td): 상태마다 입력 X, 후보별 이윤 P, 후보별 다음 상태 입력 Xn(마지막 기간은 0). 끝까지 돌리지 않아 미래 한 번의 운이 목표에 섞이지 않는다. 이미 있으면 건너뛴다
def collect_td_one(args):
  cfg, rep = args
  path = f"{DIR}/{rep}.npz"
  if os.path.exists(path):
    return rep
  t0, C = time.time(), cands(cfg)
  s, pol, X, P, Xn, T = Sim(cfg, rep), behave(cfg, rep), [], [], [], []
  while s.t < cfg.T:
    o = s.pre()
    if o.t > 0 and ready(o):
      x, pi, xn = feats(cfg, o), [], []
      for a in C:
        b = deepcopy(s)
        pi.append(b.step(*a)["Pi"])
        xn.append(feats(cfg, b.pre()) if b.t < cfg.T else np.zeros_like(x))
      X.append(x)
      P.append(pi)
      Xn.append(xn)
      T.append(o.t)
    s.step(*pol(o))
  np.savez_compressed(path, X=np.array(X, np.float32), P=np.array(P), Xn=np.array(Xn, np.float32), t=np.array(T), rep=np.full(len(T), rep))
  print(rep, len(T), round(time.time() - t0), flush=True)
  return rep


# TD 자료 (반복 순)
def load():
  D = [np.load(f) for f in sorted(glob("result/data/td/*.npz"))]
  return [np.concatenate([d[k] for d in D]) for k in ("X", "P", "Xn", "t", "rep")]


# 한 설정으로 기간 T−1부터 1까지 Q̂_t를 학습
def fqi(cfg, X, P, Xn, t, hold, name, h):
  n, k, d = X.shape
  q = QT({}, cfg.T, {}, {})
  for s in range(cfg.T - 1, 0, -1):
    i = t == s
    y = target(q if s + 1 in q.qs else None, P[i], Xn[i], cfg.eps)
    q.lo[s], q.hi[s] = float(y.min()), float(y.max())
    a, b = i & ~hold, i & hold
    Xa, ya = X[a].reshape(-1, d).astype(float), y[~hold[i]].reshape(-1)
    mu, sd = scaler(Xa)
    ys = float(ya.std()) or 1.0
    if name == "lin":
      q.qs[s] = ridge(Xa, ya, mu, sd, ys, **h)
    else:
      q.qs[s] = mlp(cfg, Xa, ya, mu, sd, ys, **h, hold=(X[b].reshape(-1, d).astype(float), y[hold[i]].reshape(-1)))
  return q


# 1차 학습 자료: 학습 반복마다 collect_one, result/data/adv.npz (X: 상태 × 후보 × 입력, Y: 상태 × 후보, rep, t, R0)
def collect():
  cfg = Cfg()
  with Pool(20) as p:
    out = p.map(collect_one, [(cfg, r) for r in range(cfg.rep0_train, cfg.rep0_train + cfg.n_train)], chunksize=1)
  os.makedirs("result/data", exist_ok=True)
  np.savez_compressed("result/data/adv.npz", **{k: np.concatenate([o[i] for o in out]) for i, k in enumerate(("X", "Y", "rep", "t", "R0"))})


# TD 학습 자료: TD 반복마다 collect_td_one
def collect_td():
  cfg = Cfg()
  os.makedirs(DIR, exist_ok=True)
  with Pool(cfg.n_proc) as p:
    p.map(collect_td_one, [(cfg, r) for r in range(cfg.rep0_td, cfg.rep0_td + cfg.n_td)], chunksize=1)


# 1차 학습: adv.npz의 학습 반복(검증 n_val개 제외)으로 선형·MLP를 학습하고 검증 점수로 고른다. 비교용으로 검증 상태에서 고정 padding(A = 0), 한 기간 최적(예상 플랫폼 몫 최대 m, θ = 0), 사후 최선 후보의 점수도 적는다
def train_qhat():
  cfg = Cfg()
  D = np.load("result/data/adv.npz")
  X, Y, rep = D["X"], D["Y"], D["rep"]
  va = rep >= cfg.rep0_train + cfg.n_train - cfg.n_val
  n, k, d = X.shape
  Xr, Yr = X[~va].reshape(-1, d), Y[~va].reshape(-1)
  mu, sd = scaler(Xr)
  ys = float(Yr.std())
  Xv, Yv = X[va], Y[va]
  G = np.where(Xv[:, :, S + 1] == 0, Xv[:, :, S + 3], -np.inf)
  ref = dict(fixed=score(np.eye(k)[np.zeros(len(Yv), int)], Yv)["val"], myopic=score(G, Yv)["val"], oracle=float(Yv.max(1).mean()))
  grid = [("lin", dict(lam=l)) for l in cfg.ridge_grid] + [("mlp", dict(L=L, H=H, lr=lr)) for L in cfg.mlp_layers for H in cfg.mlp_hidden for lr in cfg.mlp_lr]
  rows, best = [], {}
  for name, h in grid:
    t0 = time.time()
    q = ridge(Xr, Yr, mu, sd, ys, **h) if name == "lin" else mlp(cfg, Xr, Yr, mu, sd, ys, **h)
    sc = score(q(Xv.reshape(-1, d)).reshape(-1, k), Yv)
    rows.append(dict(name=name, **h, **sc, sec=time.time() - t0))
    print(rows[-1], flush=True)
    if name not in best or sc["val"] > best[name][0]["val"]:
      best[name] = (sc, h, q)
  prop = max(best, key=lambda m: best[m][0]["val"])
  pickle.dump({m: best[m][2] for m in best}, open("result/data/qhat.pkl", "wb"))
  out = dict(cfg=repr(cfg), n_train=int((~va).sum()), n_val=int(va.sum()), ref=ref, rows=rows, chosen={m: dict(**best[m][1], **best[m][0]) for m in best}, proposal=prop)
  json.dump(out, open("result/runs/train_qhat.json", "w"), indent=1)
  print(json.dumps(dict(ref=ref, chosen=out["chosen"], proposal=prop), indent=1))


# 2차 학습(TD): 기간 29부터 1까지 목표 Π + (1 − eps) Q̂_{t+1}(S', 고정) + eps 평균 Q̂_{t+1}(S', ·). 한 설정을 모든 기간에 쓰고, 학습은 조기 종료용 반복을 뺀 반복으로 하며 MLP만 그 반복의 제곱오차로 멈출 시점을 고른다. 선택 기준은 adv.npz 전체 상태에서 고른 후보의 실제 A 평균
def train_td():
  cfg = Cfg()
  X, P, Xn, t, rep = load()
  hold = rep >= cfg.rep0_td + cfg.n_td - cfg.n_hold
  A = np.load("result/data/adv.npz")
  Xv, Yv, rv = A["X"], A["Y"], A["rep"]
  sub = rv >= cfg.rep0_train + cfg.n_train - cfg.n_val
  G = np.where(Xv[:, :, 18] == 0, Xv[:, :, 20], -np.inf)
  ref = dict(fixed=0.0, myopic=score(G, Yv)["val"], oracle=float(Yv.max(1).mean()), myopic_sub=score(G[sub], Yv[sub])["val"])
  grid = [("lin", dict(lam=l)) for l in cfg.ridge_grid] + [("mlp", dict(L=L, H=H, lr=lr)) for L in cfg.mlp_layers for H in cfg.mlp_hidden for lr in cfg.mlp_lr]
  rows, best = [], {}
  for name, h in grid:
    t0 = time.time()
    q = fqi(cfg, X, P, Xn, t, hold, name, h)
    v = np.array([q(x) for x in Xv])
    sc, ss = score(v, Yv), score(v[sub], Yv[sub])
    rows.append(dict(name=name, **h, val=sc["val"], val_sub=ss["val"], pick0=float((v.argmax(1) == 0).mean()), sec=time.time() - t0))
    print(rows[-1], flush=True)
    if name not in best or sc["val"] > best[name][0]:
      best[name] = (sc["val"], h, q)
  prop = max(best, key=lambda m: best[m][0])
  pickle.dump({m: best[m][2] for m in best}, open("result/data/qhat_td.pkl", "wb"))
  out = dict(cfg=repr(cfg), n_states=len(t), n_hold=int(hold.sum()), ref=ref, rows=rows, chosen={m: dict(**best[m][1], val=best[m][0]) for m in best}, proposal=prop)
  json.dump(out, open("result/runs/train_td.json", "w"), indent=1)
  print(json.dumps(dict(ref=ref, chosen=out["chosen"], proposal=prop), indent=1))


if __name__ == "__main__":
  {"collect": collect, "collect_td": collect_td, "train_qhat": train_qhat, "train_td": train_td}[sys.argv.pop(1)]()
