"""질문 3 학습 (TD): 원재료(result.collect_raw)로 선형·MLP·MLP + 구조 요약·GNN을 같은 자료, 같은 탐색 예산 8개씩으로 기간 T−1부터 1까지 거꾸로 학습한다.
목표·조기 종료·선택 기준은 train_td와 같다 (ref/decisions.md "TD 학습 (2차)", "질문 3 비교 범위"). 평가는 adv 원재료 전체 상태에서 고른 후보의 실제 A 평균, regret, 최선 적중률, 후보 순위 상관, RMSE.
결과 result/runs/train_q3.json, 모형 result/data/qhat_q3.pkl. 실행: src/에서 python -m result.train_q3 [작업자 수]
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
import numpy as np
import torch
from utils.params import Cfg
from value.fit import QT, scaler, ridge, mlp, score
from value.gnn import gnn
from value.raw import load_adv, load_td

METHODS = ("lin", "mlp", "mlps", "gnn")
DATA = {}


# 방법별 입력: 표(lin, mlp), 표 + 구조 요약(mlps), (표, 그래프 목록)(gnn). 상태 단위 배열 → 행 단위
def inp(name, X, Xs, G):
  if name == "mlps":
    return Xs.reshape(-1, Xs.shape[-1]).astype(float)
  Z = X.reshape(-1, X.shape[-1]).astype(float)
  return (Z, [g for gs in G for g in gs]) if name == "gnn" else Z


# TD 목표: Π + (1 − eps) Q̂_{t+1}(S', 고정) + eps 평균 Q̂_{t+1}(S', ·), 다음 상태 입력은 방법에 맞게 (q가 없으면 Π)
def tgt(q, name, D, i, eps):
  P = D["P"][i]
  if q is None:
    return P
  n, k = P.shape
  Gn = [gs for j in np.flatnonzero(i) for gs in D["Gn"][j]]
  v = q(inp(name, D["Xn"][i], D["Xsn"][i], Gn)).reshape(n, k, -1)
  return P + (1 - eps) * v[:, :, 0] + eps * v.mean(2)


# 한 설정으로 기간 T−1부터 1까지 Q̂_t를 학습 (hold = 조기 종료용 상태 표시)
def fqi(cfg, D, hold, name, h):
  X, Xs, G, t = D["X"], D["Xs"], D["G"], D["t"]
  k = X.shape[1]
  q = QT({}, cfg.T, {}, {})
  for s in range(cfg.T - 1, 0, -1):
    i = t == s
    y = tgt(q if s + 1 in q.qs else None, name, D, i, cfg.eps)
    q.lo[s], q.hi[s] = float(y.min()), float(y.max())
    a, b = i & ~hold, i & hold
    ya, yb = y[~hold[i]].reshape(-1), y[hold[i]].reshape(-1)
    za, zb = inp(name, X[a], Xs[a], [G[j] for j in np.flatnonzero(a)]), inp(name, X[b], Xs[b], [G[j] for j in np.flatnonzero(b)])
    mu, sd = scaler(za[0] if name == "gnn" else za)
    ys = float(ya.std()) or 1.0
    if name == "lin":
      q.qs[s] = ridge(za, ya, mu, sd, ys, **h)
    elif name == "gnn":
      q.qs[s] = gnn(cfg, za[0], za[1], ya, mu, sd, ys, k, **h, hold=(zb[0], zb[1], yb))
    else:
      q.qs[s] = mlp(cfg, za, ya, mu, sd, ys, **h, hold=(zb, yb))
  return q


# adv 상태 전체의 후보별 예측 (기간별로 묶어 그 기간의 Q̂_t를 부름)
def predict(q, name, A):
  n, k, _ = A["X"].shape
  v = np.zeros((n, k))
  for s in np.unique(A["t"]):
    i = A["t"] == s
    v[i] = q(inp(name, A["X"][i], A["Xs"][i], [A["G"][j] for j in np.flatnonzero(i)])).reshape(-1, k)
  return v


# 상태 안 후보 순위 상관 (Spearman, 동점은 순서대로), 상수면 제외
def rho(v, Y):
  rk = lambda a: a.argsort().argsort().astype(float)
  r = [np.corrcoef(rk(a), rk(b))[0, 1] for a, b in zip(v, Y) if b.std() > 0 and a.std() > 0]
  return float(np.mean(r)) if r else 0.0


# 지표: 고른 후보의 실제 A 평균, regret, 최선 적중률, 순위 상관, RMSE, 고정 padding을 고른 비율
def metrics(v, Y):
  i = v.argmax(1)
  pick, best = Y[np.arange(len(Y)), i], Y.max(1)
  return dict(val=float(pick.mean()), regret=float((best - pick).mean()), hit=float((pick >= best - 1e-9).mean()), rho=rho(v, Y),
              rmse=float(np.sqrt(((v - Y) ** 2).mean())), pick0=float((i == 0).mean()))


# 기간 하나의 파라미터 수 (선형은 계수, 신경망은 가중치)
def n_par(q):
  return sum(p.numel() for p in q.net.parameters()) if getattr(q, "w", None) is None else int(len(q.w))


# 탐색 예산: 방법마다 8개 (선형 λ, MLP·MLP + 구조 요약·GNN은 층 × 숨은 × 학습률)
def grid(cfg):
  g = [("lin", dict(lam=l)) for l in cfg.ridge_grid]
  for name in ("mlp", "mlps", "gnn"):
    g += [(name, dict(L=L, H=H, lr=lr)) for L in cfg.mlp_layers for H in cfg.mlp_hidden for lr in cfg.mlp_lr]
  return g


# 작업자 초기화: 원재료를 한 번 읽고 torch 스레드를 제한한다
def init(cfg, th):
  torch.set_num_threads(th)
  DATA["cfg"], DATA["D"], DATA["A"] = cfg, load_td(cfg), load_adv(cfg)
  DATA["hold"] = DATA["D"]["rep"] >= cfg.rep0_td + cfg.n_td - cfg.n_hold
  DATA["sub"] = DATA["A"]["rep"] >= cfg.rep0_train + cfg.n_train - cfg.n_val


# 설정 하나 학습·평가
def fit_one(args):
  name, h = args
  cfg, D, A = DATA["cfg"], DATA["D"], DATA["A"]
  t0 = time.time()
  q = fqi(cfg, D, DATA["hold"], name, h)
  sec = time.time() - t0
  t1 = time.time()
  v = predict(q, name, A)
  row = dict(name=name, **h, **metrics(v, A["Y"]), val_sub=metrics(v[DATA["sub"]], A["Y"][DATA["sub"]])["val"], n_par=n_par(q.qs[1]), sec=sec,
             sec_pred=time.time() - t1)
  print(row, flush=True)
  return row, q


def main():
  cfg, n = Cfg(), int(sys.argv[1]) if len(sys.argv) > 1 else 8
  with Pool(n, initializer=init, initargs=(cfg, max(1, cfg.n_proc // n))) as p:
    out = p.map(fit_one, grid(cfg), chunksize=1)
  init(cfg, cfg.n_proc)
  A, sub = DATA["A"], DATA["sub"]
  Gm = np.where(A["X"][:, :, 18] == 0, A["X"][:, :, 20], -np.inf)
  ref = dict(fixed=0.0, myopic=score(Gm, A["Y"])["val"], oracle=float(A["Y"].max(1).mean()), myopic_sub=score(Gm[sub], A["Y"][sub])["val"])
  rows, best = [r for r, _ in out], {}
  for r, q in out:
    if r["name"] not in best or r["val"] > best[r["name"]][0]["val"]:
      best[r["name"]] = (r, q)
  prop = max(best, key=lambda m: best[m][0]["val"])
  pickle.dump({m: best[m][1] for m in best}, open("result/data/qhat_q3.pkl", "wb"))
  res = dict(cfg=repr(cfg), n_states=int(len(DATA["D"]["t"])), n_hold=int(DATA["hold"].sum()), n_adv=int(len(A["t"])), ref=ref, rows=rows,
             chosen={m: best[m][0] for m in best}, proposal=prop)
  json.dump(res, open("result/runs/train_q3.json", "w"), indent=1)
  print(json.dumps(dict(ref=ref, chosen=res["chosen"], proposal=prop), indent=1))


if __name__ == "__main__":
  main()
