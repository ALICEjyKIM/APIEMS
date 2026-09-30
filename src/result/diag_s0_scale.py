"""슬라이스 0 진단: 규모별 MILP 해 시간과 구조 격자의 성립 조건.
1기간 MILP(규칙 기반·근시안·가치 근사)와 H기간 hindsight optimal의 해 시간을 재고,
품목·공급자를 키운 뒤 n_alt × conc × k 격자에서 공급자 없는 품목과 충족률을 본다.
"""
import sys, json, time
from dataclasses import replace, asdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from utils.params import Cfg
from utils.instance import qty_range
from utils.common import ret_u
from utils.arrivals import new_buyers, perturb
from env.platform import Env
from env.response import Logistic
from match.milp_build import build
from match.milp_solve import Policy

ENV = gp.Env(params={"OutputFlag": 0})
OUT = "result/runs/diag_s0_scale.json"
WARM, T, REPS = 10, 30, 6
TL = 180.0
# 합성 retention value 범위 (실측 평균: 주문자 약 80, 공급자 약 900 → 상한 2배)
C_BUY, C_SUP = 160.0, 1800.0


# 품목별 안정 상태 수요와 총 공급용량
def demand(cfg):
  lo, hi = qty_range(cfg)
  dem = cfg.lam_buy / (1 - cfg.ret_ss) * cfg.items_per_order / cfg.n_items * (lo + hi) / 2
  return dem, round(cfg.cover * dem / cfg.occ)


# 규칙 기반으로 n기간 진행한 뒤의 환경
def warmed(cfg, rep, n=WARM):
  env = Env(cfg, rep)
  env.reset()
  pol = Policy(cfg, split=cfg.sh_ref)
  for _ in range(n):
    env.step(pol.act(env.o))
  return env


# 한 MILP의 빌드·해 시간과 크기
def timed(cfg, o, split, resp, c, tl=TL):
  t0 = time.perf_counter()
  m, *_ = build(cfg, o, split, resp, c)
  tb = time.perf_counter() - t0
  m.Params.TimeLimit = tl
  t0 = time.perf_counter()
  m.optimize()
  return dict(build=tb, solve=time.perf_counter() - t0, status=int(m.Status), n_var=m.NumVars,
              n_bin=m.NumBinVars, n_con=m.NumConstrs, nodes=int(m.NodeCount))


# 측정 1: 규모별 1기간 MILP 해 시간 (정책 세 종류)
def one_period(scales):
  out = []
  for ni, ns, na, k in scales:
    cfg = replace(Cfg(), n_items=ni, n_sup=ns, n_alt=na, items_per_order=k)
    env = warmed(cfg, 0)
    o, resp = env.o, Logistic(cfg)
    g = np.random.default_rng(0)
    c = (g.uniform(0, C_BUY, len(o.bid)), g.uniform(0, C_SUP, len(o.sid)))
    dem, tot = demand(cfg)
    out.append(dict(n_items=ni, n_sup=ns, n_alt=na, k=k, B=len(o.q), J=len(o.cap),
                    x_vars=len(o.q) * len(o.cap) * ni, dem=dem, cap_sup=tot / na,
                    one_covers=bool(tot / na >= dem),
                    rule=timed(cfg, o, cfg.sh_ref, resp, None),
                    myopic=timed(cfg, o, None, resp, None),
                    vfa=timed(cfg, o, None, resp, c)))
  return out


# 창 안의 고정 정보를 펼친다 (도착·변동·재참여 난수는 결정과 무관)
def unroll(cfg, env, H):
  t0, rep, inst = env.t, env.rep, env.inst
  B = [(i, 0, env.buys[i]) for i in env.o.bid]
  nb = env.nb
  for h in range(1, H):
    for m_, b in enumerate(new_buyers(cfg, inst, rep, t0 + h)):
      B.append((nb + m_, h, b))
    nb += len(new_buyers(cfg, inst, rep, t0 + h))
  S = [(env.sups[j][0], j, env.sups[j][1]) for j in env.offs]
  q = np.zeros((H, len(B), cfg.n_items), int)
  p = np.zeros((H, len(B), cfg.n_items))
  for k, (i, hb, prof) in enumerate(B):
    for h in range(hb, H):
      pr = prof if h == hb else perturb(cfg, rep, "buy", i, t0 + h, prof)
      q[h, k], p[h, k] = pr.qty, pr.price
  cap = np.zeros((H, len(S), cfg.n_items), int)
  c = np.zeros((H, len(S), cfg.n_items))
  for k, (i, j, prof) in enumerate(S):
    for h in range(H):
      pr = env.offs[j] if h == 0 else perturb(cfg, rep, "sup", i, t0 + h, prof)
      cap[h, k], c[h, k] = pr.qty, pr.price
  ub = np.array([[ret_u(cfg, rep, "buy", i, t0 + h) for h in range(H)] for i, _, _ in B])
  us = np.array([[ret_u(cfg, rep, "sup", i, t0 + h) for h in range(H)] for i, _, _ in S])
  return B, S, q, p, cap, c, ub, us, np.array([h for _, h, _ in B])


# H기간 hindsight optimal MILP: 난수를 모두 알고 배정·잉여배분을 한 번에 정한다
# 재참여는 u < p이므로 "다음 기간 활동이면 잉여율이 임계 이상"의 선형 제약이 된다. 자리 충원은 창 안에서 제외
def build_hs(cfg, unr, H):
  B, S, q, p, cap, c, ub, us, hb = unr
  nB, nS, resp = len(B), len(S), Logistic(cfg)
  m = gp.Model(env=ENV)
  m.Params.OutputFlag, m.Params.Seed = 0, cfg.grb_seed
  m.Params.Threads, m.Params.MIPGap = cfg.grb_threads, 1e-4
  x = m.addMVar((H, nB, nS, cfg.n_items), vtype=GRB.INTEGER, ub=np.minimum(q[:, :, None, :], cap[:, None, :, :]))
  y, ab = m.addMVar((H, nB), vtype=GRB.BINARY), m.addMVar((H, nB), vtype=GRB.BINARY)
  asup = m.addMVar((H, nS), vtype=GRB.BINARY)
  u, v, obj = m.addMVar((H, nB)), m.addMVar((H, nS)), 0
  for h in range(H):
    mg = (p[h][:, None] - c[h][None]) * x[h]
    m.addConstr(x[h].sum(1) == q[h] * y[h][:, None])
    m.addConstr(x[h].sum(0) <= cap[h])
    m.addConstr(u[h] <= mg.sum(2).sum(1))
    m.addConstr(v[h] <= mg.sum(2).sum(0))
    m.addConstr(y[h] <= ab[h])
    big = np.minimum(q[h][:, None, :], cap[h][None, :, :])
    m.addConstr(x[h] <= big * ab[h][:, None, None])
    m.addConstr(x[h] <= big * asup[h][None, :, None])
    prof = mg.sum() - cfg.f_ship * y[h].sum() - u[h].sum() - v[h].sum()
    m.addConstr(prof >= 0)
    obj = obj + prof
    for k in range(nB):
      m.addConstr(ab[h, k] == 0) if h < hb[k] else None
      m.addConstr(ab[h, k] == 1) if h == hb[k] else None
    if h == 0:
      m.addConstr(asup[0] == 1)
    if h + 1 < H:
      tb = np.maximum(resp.rate("buy", np.clip(ub[:, h], 1e-9, 1 - 1e-9)), 0) * (p[h] * q[h]).sum(1)
      ts = np.maximum(resp.rate("sup", np.clip(us[:, h], 1e-9, 1 - 1e-9)), 0) * (c[h] * cap[h]).sum(1)
      for k in range(nB):
        if hb[k] <= h:
          m.addConstr(ab[h + 1, k] <= ab[h, k])
          m.addConstr(u[h, k] >= tb[k] * ab[h + 1, k])
      m.addConstr(asup[h + 1] <= asup[h])
      m.addConstr(v[h] >= ts * asup[h + 1])
  m.setObjective(obj, GRB.MAXIMIZE)
  return m


# 측정 2: H기간 hindsight optimal 해 시간 (최적에 못 닿으면 그 규모에서 중단)
def hindsight(scales, hs=(1, 2, 3, 5, 8)):
  out = []
  for ni, ns, na in scales:
    cfg = replace(Cfg(), n_items=ni, n_sup=ns, n_alt=na)
    env = warmed(cfg, 0)
    for H in hs:
      unr = unroll(cfg, env, H)
      t0 = time.perf_counter()
      m = build_hs(cfg, unr, H)
      tb = time.perf_counter() - t0
      m.Params.TimeLimit = TL
      t0 = time.perf_counter()
      m.optimize()
      r = dict(n_items=ni, n_sup=ns, n_alt=na, H=H, nB=len(unr[0]), nS=len(unr[1]), build=tb,
               solve=time.perf_counter() - t0, status=int(m.Status), n_var=m.NumVars,
               n_bin=m.NumBinVars, n_con=m.NumConstrs, nodes=int(m.NodeCount),
               obj=(m.ObjVal if m.SolCount else None), gap=(m.MIPGap if m.SolCount else None))
      out.append(r)
      if r["status"] != GRB.OPTIMAL:
        break
  return out


# 측정 3: 구조 격자 칸별 성립 조건
def grid(ni, ns, alts, concs, ks):
  out = []
  for na in alts:
    for conc in concs:
      for k in ks:
        cfg = replace(Cfg(), n_items=ni, n_sup=ns, n_alt=na, conc=conc, items_per_order=k)
        dem, tot = demand(cfg)
        sup, nocov, fill, prof, tv, deg = [], [], [], [], [], None
        g = np.random.default_rng(0)
        for rep in range(REPS):
          env = Env(cfg, rep)
          env.reset()
          deg = deg or env.inst.sup_items.sum(1).tolist()
          pol = Policy(cfg, split=cfg.sh_ref)
          for t in range(T):
            o = env.o
            if t >= T // 2:
              sup.append(len(o.sid))
              nocov.append(int((o.cap.sum(0) == 0).sum()) if len(o.cap) else ni)
            if t == T - 1 and rep < 2:
              c = (g.uniform(0, C_BUY, len(o.bid)), g.uniform(0, C_SUP, len(o.sid)))
              tv.append(timed(cfg, o, None, Logistic(cfg), c)["solve"])
            r = env.step(pol.act(o))
            if t >= T // 2:
              fill.append(r["n_ok"] / max(r["n_buy"], 1))
            prof.append(r["profit"])
        out.append(dict(n_items=ni, n_sup=ns, n_alt=na, conc=conc, k=k, dem=dem, cap_item=tot,
                        cap_sup=tot / na, one_covers=bool(tot / na >= dem), deg=deg,
                        sup=float(np.mean(sup)), sup_min=int(np.min(sup)),
                        nocov=float(np.mean(nocov)), nocov_max=int(np.max(nocov)),
                        fill=float(np.mean(fill)), profit=float(np.sum(prof) / REPS),
                        vfa_sec=float(np.mean(tv))))
  return out


NOTE = ("슬라이스 0 진단 (규모 결정용). 1기간 MILP는 규칙 기반으로 10기간 진행한 뒤의 상태에서 rep 0, threads 1. "
        "가치 근사 MILP의 retention value는 합성값(주문자 U(0,160), 공급자 U(0,1800)). "
        "hindsight optimal은 H기간 단일 MILP이고 창 안의 자리 충원은 제외했다(따라서 실제 상한보다 낮다). "
        "격자는 반복 6개 × 기간 30의 후반 15기간 평균이며 occ는 기존 규모 실측값을 그대로 썼다.")

if __name__ == "__main__":
  res = dict(note=NOTE, cfg=asdict(Cfg()),
             one_period=one_period([(6, 6, 2, 2), (6, 6, 2, 3), (10, 12, 3, 2), (15, 18, 3, 2),
                                    (15, 24, 3, 2), (15, 24, 4, 2), (15, 24, 3, 3), (20, 30, 4, 2)]),
             hindsight=hindsight([(6, 6, 2), (15, 24, 4)]),
             grid=grid(15, 24, (2, 3, 4), (0.0, 0.5, 1.0), (1, 2, 3)))
  with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
  print(f"saved {OUT}")
