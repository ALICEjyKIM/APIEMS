"""상태별 이후 손실 점검 (ref/decisions.md "상태별 이후 손실 점검"): 한 번 올린 대가 점검 ×2의 250개 상태에서 입찰 전 피처가 이후 손실 L을 시점 표보다 잘 설명하는지 본다.
(1) 미래 앞 4개 · 뒤 4개 반분 신뢰도, (2) 반복 하나씩 빼는 교차 검증의 제곱오차, (3) 예상 이번 기간 이득 + 예측 L로 고른 m의 끝까지 차이(오프라인).
결과 result/runs/check_state_x2.json. 실행: src/에서 python -m result.check_state [Cfg 키=값 ...] (예: n_proc=4)
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from scipy.stats import t as tdist
from utils.params import Cfg, SCALE
from env.sim import Sim
from mech.chu import clear
from policy.base import est
from result.check_pred import REPS, T0
from result.check_dev_ci import cci
from result.exp_long_x2 import BINS

COM = ("nb", "ns", "ret_b", "ret_s", "tight", "thin")


# 반복 하나의 고정 padding 경로에서 t ≥ T0 상태마다 입찰 전 피처와 평균 예측 몫(저장값 확인용)
def path(args):
  cfg, rep = args
  s, out = Sim(cfg, rep), []
  while s.t < cfg.T:
    if s.t >= T0:
      o = s.pre()
      W, g = est(o)
      P = [clear(cfg, o.q, W, o.K, g, m) for m in cfg.m_grid]
      d = o.q.sum(0) > 0
      out.append(dict(rep=rep, t=s.t, nb=len(o.bid), ns=len(o.slot), ret_b=float(np.mean(~np.isnan(o.W0))),
                      ret_s=float(np.mean(~np.isnan(o.g0).all(1))), tight=float(o.q.sum() / o.K.sum()),
                      thin=int((o.K > 0).sum(0)[d].min()), ex=[int(p.Ub.sum() + p.Uj.sum()) for p in P], G_mean=[float(p.G) for p in P]))
    s.step(1.0, 0.0)
  return out


# 구간별 평균을 각 원소 자리에 (bb: 구간 번호)
def bm(x, bb):
  return (np.bincount(bb, x) / np.bincount(bb))[bb]


# 반복 jackknife 95% 구간: 통계 f(포함할 상태 마스크)를 반복 하나씩 빼고 다시 계산
def jack(f, g):
  u = np.unique(g)
  v = np.array([f(g != k) for k in u])
  se = np.sqrt((len(u) - 1) / len(u) * ((v - v.mean()) ** 2).sum())
  return dict(mean=float(f(np.ones(len(g), bool))), ci=float(tdist.ppf(0.975, len(u) - 1) * se))


# 최소제곱 계수
def ls(X, y):
  return np.linalg.lstsq(X, y, rcond=None)[0]


# 요약: m별 반분 신뢰도, 교차 검증 예측 오차와 계수, 고른 m의 끝까지 차이
def summ(rows):
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  B = np.eye(len(BINS))[b]
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  h, n, M = R.shape[1] // 2, len(rows), R.shape[2]
  La, Lb, D = R[:, :h].mean(1) - I, R[:, h:].mean(1) - I, R.mean(1)
  L = D - I
  Gs, Gm = (np.array([r[k] for r in rows]) for k in ("G_saa", "G_mean"))
  Gs, Gm = Gs - Gs[:, :1], Gm - Gm[:, :1]
  ex = np.array([r["ex"] for r in rows], float)
  ex = ex - ex[:, :1]
  C = np.array([[r[k] for k in COM] for r in rows], float)
  Lt, Lf, rel, pred = np.zeros((n, M)), np.zeros((n, M)), [], []
  for i in range(1, M):
    sb = lambda r: 2 * r / (1 + r)
    rc = lambda k, res: np.corrcoef(*((x[k] - bm(x[k], b[k])) if res else x[k] for x in (La[:, i], Lb[:, i])))[0, 1]
    raw, res = jack(lambda k: rc(k, False), g), jack(lambda k: rc(k, True), g)
    rel.append(dict(raw=raw, res=res, sb_raw=float(sb(raw["mean"])), sb_res=float(sb(res["mean"]))))
    F = np.c_[C, ex[:, i], Gs[:, i]]
    X = np.c_[B, F]
    for k in np.unique(g):
      tr, te = g != k, g == k
      Lt[te, i], Lf[te, i] = B[te] @ ls(B[tr], L[tr, i]), X[te] @ ls(X[tr], L[tr, i])
    sst = ((L[:, i] - L[:, i].mean()) ** 2).sum()
    Z = (F - F.mean(0)) / (F.std(0) + (F.std(0) == 0))  # 값이 하나뿐인 피처는 0 열로 둔다
    full = ls(np.c_[B, Z], L[:, i])[len(BINS):]
    same = sum(np.sign(ls(np.c_[B, Z][g != k], L[g != k, i])[len(BINS):]) == np.sign(full) for k in np.unique(g))
    pred.append(dict(r2_time=float(1 - ((L[:, i] - Lt[:, i]) ** 2).sum() / sst), r2_feat=float(1 - ((L[:, i] - Lf[:, i]) ** 2).sum() / sst),
                     gain=cci((L[:, i] - Lt[:, i]) ** 2 - (L[:, i] - Lf[:, i]) ** 2, g),
                     coef={k: float(c) for k, c in zip(COM + ("ex", "g"), full)}, same_sign={k: int(c) for k, c in zip(COM + ("ex", "g"), same)}))
  a = np.arange(n)
  pick = lambda v: D[a, v.argmax(1)]
  dec = {}
  for nm, G in (("saa", Gs), ("mean", Gm)):
    ct, cf = (G + Lt).argmax(1), (G + Lf).argmax(1)
    dec[nm] = dict(table=cci(D[a, ct], g), feat=cci(D[a, cf], g), myopic=cci(pick(G), g), feat_vs_table=cci(D[a, cf] - D[a, ct], g),
                   dev_table=float((ct > 0).mean()), dev_feat=float((cf > 0).mean()), same_choice=float((ct == cf).mean()))
  return dict(n=n, rel=rel, pred=pred, dec=dec)


def main():
  ov = dict(a.split("=") for a in sys.argv[1:])
  cfg = replace(Cfg(), **SCALE["x2"])
  cfg = replace(cfg, **{k: type(getattr(cfg, k))(v) for k, v in ov.items()})
  dev = {(r["rep"], r["t"]): r for r in json.load(open("result/runs/check_dev_x2.json"))["rows"]}
  with Pool(min(cfg.n_proc, len(REPS))) as p:
    feats = [x for out in p.map(path, [(cfg, r) for r in REPS]) for x in out]
  rows = []
  for f in feats:
    d = dev[(f["rep"], f["t"])]
    assert np.allclose(f["G_mean"], d["G_mean"])  # 다시 만든 상태가 점검 때 상태와 같다
    rows.append({**d, **{k: v for k, v in f.items() if k != "G_mean"}})
  assert len(rows) == len(dev)
  out = summ(rows)
  json.dump(dict(cfg=repr(cfg), feat=COM + ("ex", "g"), **out, rows=rows), open("result/runs/check_state_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
