"""TD 학습 (N5 2차): result/data/td/*.npz로 기간 29부터 1까지 거꾸로 행동 정책의 Q̂_t를 학습한다 (expected SARSA). 목표 = Π + (1 − eps) Q̂_{t+1}(S', 고정) + eps 평균 Q̂_{t+1}(S', ·).
선형·MLP 탐색 예산 8개씩, 한 설정을 모든 기간에 쓴다. 학습은 조기 종료용 반복을 뺀 반복으로 하고, MLP만 그 반복의 제곱오차로 멈출 시점을 고른다.
선택 기준은 1차 자료 adv.npz 전체 상태에서 고른 후보의 실제 A 평균. 결과 result/runs/train_td.json, 모형 result/data/qhat_td.pkl. 실행: src/에서 python -m result.train_td
"""
from glob import glob
import json
import pickle
import time
import numpy as np
from utils.params import Cfg
from value.fit import QT, scaler, ridge, mlp, score, target


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


def main():
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
  main()
