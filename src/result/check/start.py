"""출발 점검 (N1 · N3, 설정은 ref/decisions.md "출발 점검 설정" 이하 세 줄).
anchor: 고정 padding에서 거래한 참여자 잉여율 중앙값 (재참여 기준점), pad: 같은 입찰에서 m 후보별 빠지는 거래·플랫폼 몫, manip: 다품목 공급자 품목 간 조작 반례 탐색과 대조군, time: 기준 시장 Chu IPB 청산의 해 시간 (N1 완료 기준).
결과 result/runs/check_start_{anchor|pad|manip}[태그].json, check_time_chu.json. 실행: src/에서 python -m result.check.start {anchor|pad|manip|time} [r_buy r_sup | multi | x2 | x4]
"""
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from utils.params import Cfg, SCALE
from utils import market
from env.sim import Sim
from mech.chu import clear

REPS, T0 = range(1000, 1010), 5


# 고정 padding 경로: 기간별 입찰 (t, q, W, K, g)와 기록
def path(args):
  cfg, rep = args
  s, out = Sim(cfg, rep), []
  for _ in range(cfg.T):
    b = (s.t, s.q.copy(), s.W.copy(), s.K.copy(), s.g.copy())
    out.append((b, s.step(1.0, 0.0)))
  return out


def paths(cfg):
  with Pool(len(REPS)) as p:
    return p.map(path, [(cfg, r) for r in REPS])


# 잉여율 중앙값과 분위수 (거래한 참여자, t ≥ T0)
def anchor(cfg):
  P = paths(cfg)
  rec = [r for ps in P for _, r in ps if r["t"] >= T0]
  xb = np.concatenate([r["xb"][r["eb"]] for r in rec])
  xs = np.concatenate([r["xs"][r["es"]] for r in rec])
  q = lambda x: dict(n=len(x), med=float(np.median(x)), q10=float(np.quantile(x, 0.1)), q25=float(np.quantile(x, 0.25)), q75=float(np.quantile(x, 0.75)), q90=float(np.quantile(x, 0.9)))
  return dict(r_buy=cfg.r2_rate_buy, r_sup=cfg.r2_rate_sup, buy=q(xb), sup=q(xs),
              ret_buy=float(np.mean([r["zb"].mean() for r in rec])), ret_sup=float(np.mean([r["zs"].mean() for r in rec])),
              nb=float(np.mean([r["nb"] for r in rec])), ns=float(np.mean([r["ns"] for r in rec])))


# 한 기간 입찰을 m 후보마다 청산
def pad_one(args):
  cfg, b = args
  t, q, W, K, g = b
  rows = []
  for m in cfg.m_grid:
    o = clear(cfg, q, W, K, g, m)
    rows.append(dict(t=t, m=m, nb=len(W), win=int(o.win.sum()), Ub=int(o.Ub.sum()), Uj=int(o.Uj.sum()), ns=len(K),
                     G=float(o.G), sw=float(o.sw), sw0=float(o.sw0), Q=int(o.Q.sum()), cap=int(K.sum())))
  return rows


def pad(cfg):
  bs = [b for ps in paths(cfg) for b, r in ps if r["t"] >= T0]
  with Pool(cfg.n_proc) as p:
    res = p.map(pad_one, [(cfg, b) for b in bs])
  rows = [x for rs in res for x in rs]
  G = np.array([[r["G"] for r in rs] for rs in res])
  summ = dict(best_m_gt1=float((G.argmax(1) > 0).mean()), gain_best=float((G.max(1) - G[:, 0]).mean()))
  for m in cfg.m_grid:
    R = [r for r in rows if r["m"] == m]
    a = lambda k: float(np.mean([r[k] for r in R]))
    summ[str(m)] = dict(win_rate=float(np.mean([r["win"] / r["nb"] for r in R])), Ub=a("Ub"), Uj=a("Uj"), G=a("G"), sw=a("sw"), sw0=a("sw0"),
                        eff=float(np.sum([r["sw"] for r in R]) / np.sum([r["sw0"] for r in R])), G_share=float(np.sum([r["G"] for r in R]) / np.sum([r["sw0"] for r in R])),
                        G_neg=int(sum(r["G"] < -1e-6 for r in R)), Q_cap=float(np.mean([r["Q"] / r["cap"] for r in R])))
  return dict(n=len(bs), summary=summ, rows=rows)


# 공급자 j의 효용: Σ_c (R_jc − 실제 원가 × y_jc)
def u_sup(cfg, q, W, K, g, gt, j):
  o = clear(cfg, q, W, K, g, 1.0)
  return float(o.R[j].sum() - (gt[j] * o.y[j]).sum())


# 주문자 b의 효용: 성립하면 실제 가치 − 가격
def u_buy(cfg, q, W, K, g, Wt, b):
  o = clear(cfg, q, W, K, g, 1.0)
  return float(Wt[b] - o.p[b]) if o.win[b] else 0.0


# 한 기간에서 다품목·단일 품목 공급자와 주문자의 신고를 바꿔 효용 변화를 잰다
def manip_one(args):
  cfg, b = args
  t, q, W, K, g = b
  o, rows = clear(cfg, q, W, K, g, 1.0), []
  trade = o.y.sum(1) > 0
  nit = (K > 0).sum(1)
  for kind, js in (("multi", np.flatnonzero(trade & (nit >= 2))[:3]), ("single", np.flatnonzero(trade & (nit == 1))[:3])):
    for j in js:
      u0 = float(o.R[j].sum() - (g[j] * o.y[j]).sum())
      for c in np.flatnonzero(K[j] > 0):
        for f in (0.5, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0):
          g2 = g.copy()
          g2[j, c] *= f
          u = u_sup(cfg, q, W, K, g2, g, j)
          rows.append(dict(kind=kind, t=t, j=int(j), c=int(c), f=f, u0=u0, u=u, gain=u - u0, w=float((g[j] * K[j]).sum())))
  for bb in np.flatnonzero(o.win)[:3]:
    u0 = float(W[bb] - o.p[bb])
    for f in (0.8, 0.9, 1.1, 1.25):
      W2 = W.copy()
      W2[bb] *= f
      u = u_buy(cfg, q, W2, K, g, W, bb)
      rows.append(dict(kind="buyer", t=t, j=int(bb), c=-1, f=f, u0=u0, u=u, gain=u - u0, w=float(W[bb])))
  return rows


def manip(cfg):
  bs = [b for ps in paths(cfg) for b, r in ps if r["t"] in (10, 20)]
  with Pool(10) as p:
    rows = [x for rs in p.map(manip_one, [(cfg, b) for b in bs]) for x in rs]
  summ = {}
  for k in ("multi", "single", "buyer"):
    R = [r for r in rows if r["kind"] == k]
    pos = [r for r in R if r["gain"] > 1e-6]
    summ[k] = dict(n=len(R), agents=len({(r["t"], r["j"], r["u0"]) for r in R}), n_pos=len(pos), max_gain=max([r["gain"] for r in R], default=0.0),
                   max_rel=max([r["gain"] / r["w"] for r in R], default=0.0), mean_pos=float(np.mean([r["gain"] for r in pos])) if pos else 0.0)
  return dict(n=len(bs), summary=summ, rows=rows)


# 반복 rep의 기준 시장 입찰 (q, W, K, g): 주문자 약 20명(신규 두 기간 분량, 정상 상태 활동 주문자 수 lam_buy / (1 − ret_ss))과 기간 0 공급자
def bids(cfg, rep):
  mk = market.make(cfg, rep)
  bs = market.new_buyers(cfg, mk, rep, 0) + market.new_buyers(cfg, mk, rep, 1)
  q, W = np.array([b.qty for b in bs]), np.array([(b.qty * b.price).sum() for b in bs])
  return q, W, np.array([s.qty for s in mk.sups]), np.array([s.price for s in mk.sups])


# time: 반복 5개의 기준 시장 입찰을 m 후보마다 청산하고 해 시간을 잰다
def timing():
  cfg, rows = Cfg(), []
  for rep in range(5):
    q, W, K, g = bids(cfg, rep)
    for m in cfg.m_grid:
      t = time.perf_counter()
      o = clear(cfg, q, W, K, g, m)
      rows.append(dict(rep=rep, m=m, nb=len(W), sec=time.perf_counter() - t, milp=o.n, win=int(o.win.sum()), pairs=int((o.y > 0).sum()), G=o.G, sw=o.sw, Ub=int(o.Ub.sum()), Uj=int(o.Uj.sum())))
      print(rows[-1])
  sec = [r["sec"] for r in rows]
  out = dict(cfg=repr(cfg), rows=rows, mean_sec=float(np.mean(sec)), max_sec=float(np.max(sec)))
  json.dump(out, open("result/runs/check_time_chu.json", "w"), indent=1)
  print("mean", out["mean_sec"], "max", out["max_sec"])


# anchor · pad · manip: 하위 명령 뒤의 인자(r_buy r_sup, multi, x2, x4)로 Cfg를 정하고 결과를 저장한다
def main():
  cmd, cfg = sys.argv[1], Cfg()
  one, sc = sys.argv[2:] == ["multi"], sys.argv[2] if sys.argv[2:3] and sys.argv[2] in SCALE else None
  if one:
    cfg = replace(cfg, n_sup=cfg.n_sup_multi)
  elif sc:
    cfg = replace(cfg, **SCALE[sc], m_grid=cfg.m_check)
  elif len(sys.argv) > 2:
    cfg = replace(cfg, r2_rate_buy=float(sys.argv[2]), r2_rate_sup=float(sys.argv[3]))
  out = dict(anchor=anchor, pad=pad, manip=manip)[cmd](cfg)
  out["cfg"] = repr(cfg)
  tag = (f"_{cfg.r2_rate_buy:.4f}_{cfg.r2_rate_sup:.4f}" if cmd == "anchor" else "") + ("_multi" if one else "") + (f"_{sc}" if sc else "")
  json.dump(out, open(f"result/runs/check_start_{cmd}{tag}.json", "w"), indent=1)
  print(json.dumps(out.get("summary", {k: v for k, v in out.items() if k != "cfg"}), indent=1))


if __name__ == "__main__":
  if sys.argv[1] == "time":
    timing()
  else:
    main()
