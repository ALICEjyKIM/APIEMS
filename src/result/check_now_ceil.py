"""이번 기간 이득 예측 최대치 점검 (ref/decisions.md "이번 기간 이득 예측 최대치 점검"): 고정 padding 경로의 상태에서 누가 왔는지와 수량 · 용량은 두고 가격만 시뮬레이터 방식대로 다시 뽑아 m별로 청산한다.
다시 뽑은 이득의 평균이 참여자 유형을 안다고 칠 때의 최대치 예측이다. 재참여자는 원래 가격 × (1 + noise_price × 표준정규), 새 주문자는 U(markup_lo, markup_hi), 새 공급자는 U(cost_lo, cost_hi) × 기준가.
결과 result/runs/check_now_ceil_x2[_full].json. 실행: src/에서 python -m result.check_now_ceil [full] [Cfg 키=값 ...] (기본: t = 5 · 10 · 15 · 20 · 25, m = 1.1. full: t ≥ 5 전부, m_grid 전부)
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
from result.check_dev_ci import cci
from result.check_now_bound import cv_table
from result.check_pred import T0
from result.exp_long_x2 import BINS

REPS, NR = range(1000, 1010), 10


# 가격만 다시 뽑은 입찰 (W, g): 재참여자는 원래 가격에 변동, 이번 기간 새로 온 참여자는 분포에서 새로
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


# 반복 하나의 경로에서 ts 상태마다 실제 이득과 다시 뽑은 이득들 (m별)
def path(args):
  cfg, rep, ts, ms = args
  s, out = Sim(cfg, rep), []
  while s.t <= max(ts):
    if s.t in ts:
      out.append(dict(rep=rep, t=s.t, act=gains(cfg, s, s.W, s.g, ms), re=[gains(cfg, s, *redraw(cfg, s, r), ms) for r in range(NR)]))
    s.step(1.0, 0.0)
  return out


# m별 상관 · 분산 분해, 고른 m의 끝까지 차이 (m = 1과 ms 중에서, 교차 검증 이후 손실 표)
def summ(rows, dev, ms, grid):
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


def main():
  full = "full" in sys.argv[1:]
  ov = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
  cfg = replace(Cfg(), **SCALE["x2"])
  cfg = replace(cfg, **{k: type(getattr(cfg, k))(v) for k, v in ov.items()})
  ts, ms = (tuple(range(T0, cfg.T)), tuple(cfg.m_grid[1:])) if full else ((5, 10, 15, 20, 25), (1.1,))
  with Pool(min(cfg.n_proc, len(REPS))) as p:
    rows = [x for out in p.map(path, [(cfg, r, ts, ms) for r in REPS]) for x in out]
  out = summ(rows, json.load(open("result/runs/check_state_x2.json"))["rows"], ms, cfg.m_grid)
  name = "result/runs/check_now_ceil_x2" + ("_full" if full else "") + ".json"
  json.dump(dict(cfg=repr(cfg), reps=list(REPS), ts=ts, nr=NR, ms=ms, **out, rows=rows), open(name, "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
