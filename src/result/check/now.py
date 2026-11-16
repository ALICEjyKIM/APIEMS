"""이번 기간 이득 예측 점검 넷 (ref/decisions.md "이번 기간 이득 추가 이득 상한", "이번 기간 이득 예측 점검", "이번 기간 이득 예측 최대치 점검", "지난 기간 이득 피처 점검"). 이후 손실 표는 반복 하나씩 빼고 만든 기간 구간 평균(교차 검증)을 모든 비교에 똑같이 쓴다.
bound: 예상 이번 기간 이득을 실제 이득으로 바꿔 고른 m의 끝까지 차이. pred: 입찰 전 정보로 실제 이번 기간 이득을 지금 예측보다 잘 맞히는지. lag: pred에 지난 기간 실제 이득 피처를 더한다. ceil: 가격만 시뮬레이터 방식대로 다시 뽑아 참여자 유형을 안다고 칠 때의 예측 최대치.
결과 result/runs/check_now_{bound|pred|lag|ceil}_x2.json (ceil full은 check_now_ceil_x2_full.json). 실행: src/에서 python -m result.check.now bound|pred|lag|ceil (ceil만 [full] [Cfg 키=값 ...]. 기본 t = 5 · 10 · 15 · 20 · 25, m = 1.1, full은 t ≥ 5 전부 · m_grid 전부)
"""
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from utils.rng import rng
from env.sim import Sim
from mech.chu import clear
from result.common import BINS, COM, T0, cci, ls

REPS, NR = range(1000, 1010), 10


# 기간 구간별 평균 이후 손실 (반복 하나씩 빼고 계산해 그 반복 상태에 붙인다)
def cv_table(L, b, g):
  out = np.zeros_like(L)
  for k in np.unique(g):
    for j in np.unique(b):
      out[(g == k) & (b == j)] = L[(g != k) & (b == j)].mean(0)
  return out


# 반복 하나씩 빼고 X로 y를 맞춘 예측
def cv_fit(X, y, g):
  p = np.zeros(len(y))
  for k in np.unique(g):
    p[g == k] = X[g == k] @ ls(X[g != k], y[g != k])
  return p


# 예측 지표: 상관, 교차 검증 R²
def score(p, y):
  return dict(corr=float(np.corrcoef(p, y)[0, 1]), r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


# 가격만 다시 뽑은 입찰 (W, g): 재참여자는 원래 가격 × (1 + noise_price × 표준정규), 이번 기간 새로 온 주문자는 U(markup_lo, markup_hi), 공급자는 U(cost_lo, cost_hi) × 기준가
def redraw(cfg, s, r):
  u, base = rng(cfg, s.rep, "redraw", s.t, r), s.mk.base
  W = np.array([(b["cur"].qty * (b["prof"].price * (1 + cfg.noise_price * u.standard_normal(cfg.n_items)) if b["ten"] else
                                 base * u.uniform(cfg.markup_lo, cfg.markup_hi) * b["prof"].items)).sum() for b in (s.B[i] for i in s.bid)])
  g = np.array([(p["prof"].price * (1 + cfg.noise_price * u.standard_normal(cfg.n_items)) if p["ten"] else
                 base * u.uniform(cfg.cost_lo, cfg.cost_hi, cfg.n_items) * p["prof"].items) for p in (s.S[j] for j in s.slot)]).reshape(-1, cfg.n_items)
  return W, np.where(s.K > 0, g, 0.0)


# 입찰 (W, g)에서 m = 1 대비 각 m의 플랫폼 몫 차이
def gains(cfg, s, W, g, ms):
  g1 = clear(cfg, s.q, W, s.K, g, 1.0).G
  return [float(clear(cfg, s.q, W, s.K, g, m).G - g1) for m in ms]


# 반복 하나의 고정 padding 경로에서 ts 상태마다 실제 이득과 다시 뽑은 이득들 (m별)
def path(args):
  cfg, rep, ts, ms = args
  s, out = Sim(cfg, rep), []
  while s.t <= max(ts):
    if s.t in ts:
      out.append(dict(rep=rep, t=s.t, act=gains(cfg, s, s.W, s.g, ms), re=[gains(cfg, s, *redraw(cfg, s, r), ms) for r in range(NR)]))
    s.step(1.0, 0.0)
  return out


# ceil 요약: m별 상관 · 분산 분해, 고른 m의 끝까지 차이 (m = 1과 ms 중에서, 교차 검증 이후 손실 표)
def summ_ceil(rows, dev, ms, grid):
  key = {(r["rep"], r["t"]): i for i, r in enumerate(dev)}
  g, t = np.array([r["rep"] for r in dev]), np.array([r["t"] for r in dev])
  I, R = np.array([r["I"] for r in dev]), np.array([r["R"] for r in dev])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, np.searchsorted(BINS, t, side="right") - 1, g)
  ix, col = np.array([key[(r["rep"], r["t"])] for r in rows]), [list(grid).index(m) for m in ms]
  act, re = np.array([r["act"] for r in rows]), np.array([r["re"] for r in rows])
  assert np.allclose(act, I[ix][:, col])  # 다시 만든 상태의 실제 이득이 저장값과 같다
  ceil, Gs = re.mean(1), np.array([[dev[i]["G_saa"][c] - dev[i]["G_saa"][0] for c in col] for i in ix])
  by_m = []
  for k, m in enumerate(ms):
    vb, vw, vc = act[:, k].var(ddof=1), re[:, :, k].var(1, ddof=1).mean(), ceil[:, k].var(ddof=1)
    by_m.append(dict(m=m, corr_ceil=float(np.corrcoef(ceil[:, k], act[:, k])[0, 1]), corr_saa=float(np.corrcoef(Gs[:, k], act[:, k])[0, 1]),
                     var_between=float(vb), var_within=float(vw), var_ceil=float(vc), pred_share=float((vc - vw / re.shape[1]) / vb),
                     mean_gap=cci(act[:, k] - ceil[:, k], g[ix])))
  Dm, Lm, gg = D[ix][:, col], Lt[ix][:, col], g[ix]
  pick = lambda v: np.where((v + Lm).max(1) > 0, Dm[np.arange(len(ix)), (v + Lm).argmax(1)], 0.0)
  dec = dict(ceil=cci(pick(ceil), gg), saa=cci(pick(Gs), gg), now=cci(pick(act), gg), ceil_vs_saa=cci(pick(ceil) - pick(Gs), gg))
  return dict(n=len(rows), by_m=by_m, dec=dec)


# pred 요약: m = 1.1 · 1.25 · 1.5마다 따로 반복 하나씩 빼는 교차 검증. 비교: 지금 예측(시나리오 · 평균 그대로), 시나리오 선형 보정, 피처 회귀, 실제 이득(상한)
def summ_pred(rows):
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  B = np.eye(len(BINS))[b]
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, b, g)
  Gs, Gm = (np.array([r[k] for r in rows]) for k in ("G_saa", "G_mean"))
  Gs, Gm = Gs - Gs[:, :1], Gm - Gm[:, :1]
  ex = np.array([r["ex"] for r in rows], float)
  ex = ex - ex[:, :1]
  C = np.array([[r[k] for k in COM] for r in rows], float)
  n, M = I.shape
  P = {k: np.zeros((n, M)) for k in ("cal", "reg")}
  pred = []
  for i in range(1, M):
    one = np.ones((n, 1))
    P["cal"][:, i] = cv_fit(np.c_[one, Gs[:, i]], I[:, i], g)
    P["reg"][:, i] = cv_fit(np.c_[B, Gs[:, i], Gm[:, i], C, ex[:, i]], I[:, i], g)
    e = lambda p: (I[:, i] - p) ** 2
    pred.append(dict(saa=score(Gs[:, i], I[:, i]), mean=score(Gm[:, i], I[:, i]), cal=score(P["cal"][:, i], I[:, i]), reg=score(P["reg"][:, i], I[:, i]),
                     gain=cci(e(P["cal"][:, i]) - e(P["reg"][:, i]), g)))
  a = np.arange(n)
  ch = {k: (v + Lt).argmax(1) for k, v in dict(saa=Gs, mean=Gm, cal=P["cal"], reg=P["reg"], now=I).items()}
  dec = {k: dict(D=cci(D[a, c], g), dev=float((c > 0).mean())) for k, c in ch.items()}
  dec["reg_vs_saa"], dec["reg_vs_cal"] = cci(D[a, ch["reg"]] - D[a, ch["saa"]], g), cci(D[a, ch["reg"]] - D[a, ch["cal"]], g)
  dec["closed"] = float((dec["reg"]["D"]["mean"] - dec["saa"]["D"]["mean"]) / (dec["now"]["D"]["mean"] - dec["saa"]["D"]["mean"]))
  return dict(n=n, pred=pred, dec=dec)


# bound: 한 번 올린 대가 점검 ×2의 250개 상태에서 기간별 평균 이후 손실 반영 정책의 예상 이번 기간 이득을 실제 이득으로 바꿔 고른 m의 끝까지 차이. 비교: 실제 이득 + 표, 시나리오 예측 + 표, 평균 예측 + 표, 실제 이득만(참고), 고정 padding 정책(0)
def bound():
  rows = json.load(open("result/runs/check_dev_x2.json"))["rows"]
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, b, g)
  Gs, Gm = (np.array([r[k] for r in rows]) for k in ("G_saa", "G_mean"))
  Gs, Gm = Gs - Gs[:, :1], Gm - Gm[:, :1]
  a = np.arange(len(rows))
  ch = {k: v.argmax(1) for k, v in dict(now=I + Lt, saa=Gs + Lt, mean=Gm + Lt, now_only=I).items()}
  out = {k: dict(D=cci(D[a, c], g), dev=float((c > 0).mean()), m_count=np.bincount(c, minlength=D.shape[1]).tolist()) for k, c in ch.items()}
  out["now_vs_saa"], out["now_vs_mean"] = cci(D[a, ch["now"]] - D[a, ch["saa"]], g), cci(D[a, ch["now"]] - D[a, ch["mean"]], g)
  out["corr_saa"] = [float(np.corrcoef(Gs[:, i], I[:, i])[0, 1]) for i in range(1, D.shape[1])]
  out["corr_mean"] = [float(np.corrcoef(Gm[:, i], I[:, i])[0, 1]) for i in range(1, D.shape[1])]
  json.dump(dict(n=len(rows), bins=BINS, **out), open("result/runs/check_now_bound_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


# pred: check_state_x2.json의 250개 상태에서 입찰 전 정보로 실제 이번 기간 이득을 지금 예측보다 잘 맞히는지, 그 예측으로 고른 m이 더 버는지
def pred():
  out = summ_pred(json.load(open("result/runs/check_state_x2.json"))["rows"])
  json.dump(out, open("result/runs/check_now_pred_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


# lag: 지난 기간 값이 있는 t ≥ 6 상태에서 m마다 반복 하나씩 빼는 교차 검증. 예측: 시나리오 그대로, 선형 보정, 시나리오 + lag1, 시나리오 + 최근 3기간 평균(lag3), 피처 회귀 + lag1(참고)
def lag():
  rows = json.load(open("result/runs/check_state_x2.json"))["rows"]
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, b, g)
  Gs = np.array([r["G_saa"] for r in rows])
  Gs = Gs - Gs[:, :1]
  ex = np.array([r["ex"] for r in rows], float)
  ex = ex - ex[:, :1]
  C = np.array([[r[k] for k in COM] for r in rows], float)
  at = {(r["rep"], r["t"]): i for i, r in enumerate(rows)}
  keep = np.array([(r["rep"], r["t"] - 1) in at for r in rows])
  lag1 = np.array([I[at[(r["rep"], r["t"] - 1)]] if k else I[i] * np.nan for i, (r, k) in enumerate(zip(rows, keep))])
  lag3 = np.array([np.mean([I[at[(r["rep"], s)]] for s in range(r["t"] - 3, r["t"]) if (r["rep"], s) in at], 0) if k else I[i] * np.nan
                   for i, (r, k) in enumerate(zip(rows, keep))])
  g, b, I, D, Lt, Gs, ex, C, lag1, lag3 = (x[keep] for x in (g, b, I, D, Lt, Gs, ex, C, lag1, lag3))
  B = np.eye(len(BINS))[b]
  n, M = I.shape
  one = np.ones((n, 1))
  P = {k: np.zeros((n, M)) for k in ("cal", "lag1", "lag3", "reg_lag")}
  pred = []
  for i in range(1, M):
    P["cal"][:, i] = cv_fit(np.c_[one, Gs[:, i]], I[:, i], g)
    P["lag1"][:, i] = cv_fit(np.c_[one, Gs[:, i], lag1[:, i]], I[:, i], g)
    P["lag3"][:, i] = cv_fit(np.c_[one, Gs[:, i], lag3[:, i]], I[:, i], g)
    P["reg_lag"][:, i] = cv_fit(np.c_[B, Gs[:, i], C, ex[:, i], lag1[:, i]], I[:, i], g)
    e = lambda p: (I[:, i] - p) ** 2
    pred.append(dict(saa=score(Gs[:, i], I[:, i]), **{k: score(P[k][:, i], I[:, i]) for k in P},
                     corr_lag1=float(np.corrcoef(lag1[:, i], I[:, i])[0, 1]), corr_lag3=float(np.corrcoef(lag3[:, i], I[:, i])[0, 1]),
                     gain_lag1=cci(e(P["cal"][:, i]) - e(P["lag1"][:, i]), g), gain_lag3=cci(e(P["cal"][:, i]) - e(P["lag3"][:, i]), g)))
  a = np.arange(n)
  ch = {k: (v + Lt).argmax(1) for k, v in dict(saa=Gs, now=I, **P).items()}
  dec = {k: dict(D=cci(D[a, c], g), dev=float((c > 0).mean())) for k, c in ch.items()}
  for k in ("lag1", "lag3", "reg_lag"):
    dec[k + "_vs_saa"] = cci(D[a, ch[k]] - D[a, ch["saa"]], g)
    dec[k + "_closed"] = float((dec[k]["D"]["mean"] - dec["saa"]["D"]["mean"]) / (dec["now"]["D"]["mean"] - dec["saa"]["D"]["mean"]))
  out = dict(n=n, pred=pred, dec=dec)
  json.dump(out, open("result/runs/check_now_lag_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


# ceil: 고정 padding 경로의 상태에서 누가 왔는지와 수량 · 용량은 두고 가격만 다시 뽑아 m별로 청산한다. 다시 뽑은 이득의 평균이 참여자 유형을 안다고 칠 때의 최대치 예측이다
def ceil():
  full = "full" in sys.argv[1:]
  ov = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
  cfg = replace(Cfg(), **SCALE["x2"])
  cfg = replace(cfg, **{k: type(getattr(cfg, k))(v) for k, v in ov.items()})
  ts, ms = (tuple(range(T0, cfg.T)), tuple(cfg.m_grid[1:])) if full else ((5, 10, 15, 20, 25), (1.1,))
  with Pool(min(cfg.n_proc, len(REPS))) as p:
    rows = [x for out in p.map(path, [(cfg, r, ts, ms) for r in REPS]) for x in out]
  out = summ_ceil(rows, json.load(open("result/runs/check_state_x2.json"))["rows"], ms, cfg.m_grid)
  name = "result/runs/check_now_ceil_x2" + ("_full" if full else "") + ".json"
  json.dump(dict(cfg=repr(cfg), reps=list(REPS), ts=ts, nr=NR, ms=ms, **out, rows=rows), open(name, "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  {"bound": bound, "pred": pred, "lag": lag, "ceil": ceil}[sys.argv.pop(1)]()
