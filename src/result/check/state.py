"""상태별 이후 손실 점검 (ref/decisions.md "상태별 이후 손실 점검", "상태 신호 점검 (작은 시장)"): 입찰 전 피처가 이후 손실 L을 기간 구간 평균보다 잘 설명하는지 본다.
기본: 한 번 올린 대가 점검 ×2의 250개 상태에서 (1) 미래 앞 4개 · 뒤 4개 반분 신뢰도, (2) 반복 하나씩 빼는 교차 검증의 제곱오차, (3) 예상 이번 기간 이득 + 예측 L로 고른 m의 끝까지 차이(오프라인). rel: 제외 연쇄 점검 결과로 m마다 미래 앞 절반 · 뒤 절반으로 잰 L의 상관(기간 구간 평균을 뺀 값 · 빼지 않은 값), 미래 전체 평균의 신뢰도(Spearman-Brown), 신뢰도 0.5에 필요한 미래 수, 기준 시장 대비 상관 차이.
결과 result/runs/check_state_x2.json, check_rel_x1.json. 실행: src/에서 python -m result.check.state [Cfg 키=값 ...] (예: n_proc=4), python -m result.check.state rel base het1.0 ... (태그마다 check_chain_x1_{태그}.json, 첫 태그가 기준)
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import Sim
from mech.chu import clear
from policy.base import est
from result.common import REPS, T0, BINS, COM, cci, jack, ls


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


# rel: m별 반분 상관(raw · res), 미래 전체 평균의 신뢰도 sb, 신뢰도 0.5에 필요한 미래 수 need
def rel(rows):
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I = I.mean(1) if I.ndim == 3 else I  # 제외 연쇄 점검은 I를 미래마다 같은 값으로 저장
  I, R = I - I[:, :1], R - R[:, :, :1]
  h, out = R.shape[1] // 2, []
  La, Lb = R[:, :h].mean(1) - I, R[:, h:].mean(1) - I
  for i in range(1, R.shape[2]):
    rc = lambda k, res: np.corrcoef(*((x[k] - bm(x[k], b[k])) if res else x[k] for x in (La[:, i], Lb[:, i])))[0, 1]
    raw, res = jack(lambda k: rc(k, False), g), jack(lambda k: rc(k, True), g)
    sb = 2 * res["mean"] / (1 + res["mean"])
    out.append(dict(raw=raw, res=res, sb=float(sb), need=float(R.shape[1] * (1 - sb) / sb) if sb > 0 else None))
  return out


# rel 실행: 태그마다 check_chain_x1_{태그}.json의 신뢰도, 첫 태그 대비 차이
def rel_main():
  tags = sys.argv[1:]
  R = {k: rel(json.load(open(f"result/runs/check_chain_x1_{k}.json"))["rows"]) for k in tags}
  for k in tags[1:]:
    for a, z in zip(R[k], R[tags[0]]):
      a["vs_base"] = dict(mean=a["res"]["mean"] - z["res"]["mean"], ci=float(np.hypot(a["res"]["ci"], z["res"]["ci"])))
  out = dict(base=tags[0], ms=list(Cfg().m_grid[1:]), rel=R)
  json.dump(out, open("result/runs/check_rel_x1.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  if sys.argv[1:2] == ["rel"]:
    sys.argv.pop(1)
    rel_main()
  else:
    main()
