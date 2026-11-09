"""이번 기간 이득 예측 최대치 점검 (ref/decisions.md "이번 기간 이득 예측 최대치 점검"): 고정 padding 경로의 상태에서 누가 왔는지와 수량 · 용량은 두고 가격만 시뮬레이터 방식대로 다시 뽑아 m = 1 · 1.1로 청산한다.
다시 뽑은 이득의 평균이 참여자 유형을 안다고 칠 때의 최대치 예측이다. 재참여자는 원래 가격 × (1 + noise_price × 표준정규), 새 주문자는 U(markup_lo, markup_hi), 새 공급자는 U(cost_lo, cost_hi) × 기준가.
결과 result/runs/check_now_ceil_x2.json. 실행: src/에서 python -m result.check_now_ceil [Cfg 키=값 ...] (예: n_proc=4)
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
from result.exp_long_x2 import BINS

REPS, TS, NR, M1 = range(1000, 1010), (5, 10, 15, 20, 25), 10, 1.1


# 가격만 다시 뽑은 입찰 (W, g): 재참여자는 원래 가격에 변동, 이번 기간 새로 온 참여자는 분포에서 새로
def redraw(cfg, s, r):
  u, base = rng(cfg, s.rep, "redraw", s.t, r), s.mk.base
  W = np.array([(b["cur"].qty * (b["prof"].price * (1 + cfg.noise_price * u.standard_normal(cfg.n_items)) if b["ten"] else
                                 base * u.uniform(cfg.markup_lo, cfg.markup_hi) * b["prof"].items)).sum() for b in (s.B[i] for i in s.bid)])
  g = np.array([(p["prof"].price * (1 + cfg.noise_price * u.standard_normal(cfg.n_items)) if p["ten"] else
                 base * u.uniform(cfg.cost_lo, cfg.cost_hi, cfg.n_items) * p["prof"].items) for p in (s.S[j] for j in s.slot)]).reshape(-1, cfg.n_items)
  return W, np.where(s.K > 0, g, 0.0)


# 반복 하나의 경로에서 TS 상태마다 실제 m = 1.1 이득과 다시 뽑은 이득들
def path(args):
  cfg, rep = args
  s, out = Sim(cfg, rep), []
  while s.t <= max(TS):
    if s.t in TS:
      act = clear(cfg, s.q, s.W, s.K, s.g, M1).G - clear(cfg, s.q, s.W, s.K, s.g, 1.0).G
      re = []
      for r in range(NR):
        W, g = redraw(cfg, s, r)
        re.append(clear(cfg, s.q, W, s.K, g, M1).G - clear(cfg, s.q, W, s.K, g, 1.0).G)
      out.append(dict(rep=rep, t=s.t, act=float(act), re=[float(x) for x in re]))
    s.step(1.0, 0.0)
  return out


def summ(rows, dev):
  key = {(r["rep"], r["t"]): i for i, r in enumerate(dev)}
  g = np.array([r["rep"] for r in dev])
  t = np.array([r["t"] for r in dev])
  I, R = np.array([r["I"] for r in dev]), np.array([r["R"] for r in dev])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, np.searchsorted(BINS, t, side="right") - 1, g)
  ix = np.array([key[(r["rep"], r["t"])] for r in rows])
  act, re = np.array([r["act"] for r in rows]), np.array([r["re"] for r in rows])
  assert np.allclose(act, I[ix, 1])  # 다시 만든 상태의 실제 이득이 저장값과 같다
  ceil, Gs = re.mean(1), np.array([dev[i]["G_saa"][1] - dev[i]["G_saa"][0] for i in ix])
  gg, d1, l1 = g[ix], D[ix, 1], Lt[ix, 1]
  pick = lambda v: np.where(v + l1 > 0, d1, 0.0)
  return dict(n=len(rows), corr_ceil=float(np.corrcoef(ceil, act)[0, 1]), corr_saa=float(np.corrcoef(Gs, act)[0, 1]),
              var_between=float(act.var(ddof=1)), var_within=float(re.var(1, ddof=1).mean()), var_ceil=float(ceil.var(ddof=1)),
              dec=dict(ceil=cci(pick(ceil), gg), saa=cci(pick(Gs), gg), now=cci(pick(act), gg),
                       ceil_vs_saa=cci(pick(ceil) - pick(Gs), gg), dev_ceil=float((ceil + l1 > 0).mean()), dev_saa=float((Gs + l1 > 0).mean())))


def main():
  ov = dict(a.split("=") for a in sys.argv[1:])
  cfg = replace(Cfg(), **SCALE["x2"])
  cfg = replace(cfg, **{k: type(getattr(cfg, k))(v) for k, v in ov.items()})
  with Pool(min(cfg.n_proc, len(REPS))) as p:
    rows = [x for out in p.map(path, [(cfg, r) for r in REPS]) for x in out]
  out = summ(rows, json.load(open("result/runs/check_state_x2.json"))["rows"])
  json.dump(dict(cfg=repr(cfg), reps=list(REPS), ts=TS, nr=NR, m=M1, **out, rows=rows), open("result/runs/check_now_ceil_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
