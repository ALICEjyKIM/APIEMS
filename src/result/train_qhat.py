"""Q̂ 학습 (N5): result/data/adv.npz의 학습 반복(검증 n_val개 제외)으로 선형·MLP를 탐색 예산 8개씩 학습하고 검증 점수로 고른다.
비교용으로 검증 상태에서 고정 padding(A = 0), 한 기간 최적(예상 플랫폼 몫 최대 m, θ = 0), 사후 최선 후보의 점수도 적는다.
결과는 result/runs/train_qhat.json, 고른 모형은 result/data/qhat.pkl. 실행: src/에서 python -m result.train_qhat
"""
import json
import pickle
import time
import numpy as np
from utils.params import Cfg
from value.fit import scaler, ridge, mlp, score

S, A = 17, 11  # 입력의 상태·후보 열 수 (value/feat.py)


def main():
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


if __name__ == "__main__":
  main()
