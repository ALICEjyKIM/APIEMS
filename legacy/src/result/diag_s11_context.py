"""진단 s11: 공급자 c_i가 own feature를 넘어 network context에 따라 달라지는가 (PROC15H, Rule-continuation 값, 정책 성능 아님).
상태는 Myopic·Rule trajectory에서 같은 수로 모으고, 공급자마다 c_i = V(S) − V(S∖i)를 같은 미래 키(CRN)로 잰다.
일부 (상태, 공급자)에는 own feature를 그대로 두고 주변을 하나씩 바꾼 상태에서 c_i를 다시 잰다. 단위 결과는 runs/diag_s11_units/.
"""
import copy
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from env.platform import Env
from match.milp_solve import Policy
from bench.rule_split import rule
from utils.params import Cfg, PROC15H
from utils.instance import qty_range
from result import diag

UNITS = diag.RUNS / "diag_s11_units"
PLAN = diag.RUNS / "diag_s11_plan.json"
CFG = replace(Cfg(), **PROC15H)
POLS = {"Myopic": lambda c: Policy(c), "Rule": lambda c: rule(c)}
REPS = range(8)
TS = (6, 12, 18)
M = 40
KEY0 = 5_000_000
N_PERT = 24
DEM_HI, DEM_LO = 1.5, 0.5
SLOW = 2
WORKERS = 12


# 미래 키 (반복, 기간, k): 실험 5~7, 진단 s9·s10 키와 겹치지 않는 영역
def key(cfg, rep, t, k):
  return KEY0 + (rep * cfg.T + t) * 100 + k


# 정책 pol로 반복 rep을 기간 t까지 걸은 상태
def state(pol, rep, t):
  env, p = Env(CFG, rep), POLS[pol](CFG)
  env.reset()
  for _ in range(t):
    env.step(p.act(env.o))
  return env


# 공급자 ID 집합 drop을 뺀 상태의 미래 M개 Rule 이윤 합 (같은 키라 상태·drop만 다르다)
def value(env, drop):
  cfg, out = env.cfg, []
  for k in range(M):
    e = copy.deepcopy(env)
    e.rep = key(cfg, env.rep, env.t, k)
    for s in drop:
      e.drop("sup", e.o.sid.index(s))
    pol = rule(cfg)
    out.append(sum(e.step(pol.act(e.o))["profit"] for _ in range(cfg.vf_H)))
  return out


# 공급자 k의 own feature와 network context feature (관측 o, 대기열 pipe)
def feats(env, k):
  o, cfg = env.o, env.cfg
  have = o.cap > 0
  it = have[k]
  deg, capi, dem = have.sum(0), o.cap.sum(0), o.q.sum(0)
  lvl = o.c[k][it] / env.inst.base[it]
  sid = o.sid[k]
  own = dict(cap=float(o.cap[k].sum()), n_items=int(it.sum()), cost_mean=float(lvl.mean()), cost_min=float(lvl.min()),
             hs=float(o.hs[k]), rs=float(o.rs[k]), tenure=int(o.t - sid // cfg.n_sup) if sid >= cfg.n_sup else int(o.t))
  alt = deg[it] - 1
  altcap = capi[it] - o.cap[k][it]
  ratio = dem[it] / np.maximum(capi[it], 1)
  conn = np.where((o.q[:, it] > 0).any(1))[0]
  bmin = [int(deg[o.q[b] > 0].min()) for b in conn]
  rel = [j for j in env.pipe if (env.pipe[j][1].qty[it] > 0).any()]
  ctx = dict(alt_min=int(alt.min()), alt_mean=float(alt.mean()), altcap_min=float(altcap.min()), altcap_mean=float(altcap.mean()),
             share_max=float((o.cap[k][it] / capi[it]).max()), dem_sum=float(dem[it].sum()), ratio_max=float(ratio.max()),
             ratio_mean=float(ratio.mean()), hi_max=float(o.hi[it].max()), hi_mean=float(o.hi[it].mean()),
             n_buy=len(conn), buy_dem=float(o.q[conn].sum()),
             hb_mean=float(o.hb[conn].mean()) if len(conn) else cfg.h0, hb_min=float(o.hb[conn].min()) if len(conn) else cfg.h0,
             bundle_min_deg=float(np.mean(bmin)) if bmin else float(deg[it].min()), bundle_bottleneck=int(sum(m <= 2 for m in bmin)),
             pend_cap=float(sum(env.pipe[j][1].qty[it].sum() for j in rel)), n_pend=len(rel),
             wait_min=int(min(env.pipe[j][0] - o.t for j in rel)) if rel else cfg.onboard + 1,
             zero_if_removed=int((alt == 0).any()), deg_min=int(deg[it].min()))
  return own, ctx


# 섭동: own feature는 그대로 두고 주변 하나만 바꾼다 (적용할 수 없으면 None)
# alt_minus: i의 가장 희소한 품목을 함께 공급하는 다른 공급자 한 명 제거 (그 자리는 onboarding 대기로 감)
# alt_plus: i의 품목을 공급하는 대기 후보 한 명을 지금 활동시킴 (자리 수가 고정이라 공급자를 더하는 유일한 현실적 경로)
# slow: i의 품목을 공급하는 대기 후보들의 활동 시점을 SLOW기간 늦춤
# dem_hi / dem_lo: i의 품목을 원하는 주문자들의 품목별 수량을 DEM_HI / DEM_LO배 (생성 범위 [lo, hi]로 자름, 프로필도 같이)
def perturb(env, sid, case):
  o, cfg = env.o, env.cfg
  k = o.sid.index(sid)
  it = o.cap[k] > 0
  if case == "alt_minus":
    deg = (o.cap > 0).sum(0)
    item = np.where(it)[0][np.argmin(deg[it])]
    alts = [o.sid[j] for j in range(len(o.sid)) if j != k and o.cap[j, item] > 0]
    if not alts:
      return None
    env.drop("sup", o.sid.index(max(alts, key=lambda s: o.cap[o.sid.index(s), item])))
    return dict(item=int(item))
  rel = sorted(j for j in env.pipe if (env.pipe[j][1].qty[it] > 0).any())
  if case == "alt_plus":
    if not rel:
      return None
    j = max(rel, key=lambda j: env.pipe[j][1].qty[it].sum())
    _, prof = env.pipe.pop(j)
    env.sups = dict(sorted({**env.sups, j: (env.t * cfg.n_sup + j, prof)}.items()))
    env.offs = {s: env.offs[s] if s in env.offs else prof for s in env.sups}
    env._observe()
    return dict(seat=int(j))
  if case == "slow":
    if not rel:
      return None
    for j in rel:
      env.pipe[j] = (env.pipe[j][0] + SLOW, env.pipe[j][1])
    env._observe()
    return dict(seats=[int(j) for j in rel])
  f = DEM_HI if case == "dem_hi" else DEM_LO
  lo, hi = qty_range(cfg)
  bs = [b for b in env.ords if (env.ords[b].qty[it] > 0).any()]
  if not bs:
    return None
  for b in bs:
    for prof in (env.ords[b], env.buys[b]):
      prof.qty = np.where(prof.qty > 0, np.clip(np.round(prof.qty * f), lo, hi), 0).astype(int)
  env._observe()
  return dict(buyers=len(bs))


CASES = ("alt_minus", "alt_plus", "slow", "dem_hi", "dem_lo")


# 작업 단위: (정책, 반복, 기간, 섭동 (공급자, case) 또는 None, 묶음 번호, 제거 집합들)
def unit(job):
  pol, rep, t, pert, c, drops = job
  name = f"{pol}_r{rep}_t{t}" + (f"_{pert[1]}_s{pert[0]}" if pert else f"_c{c}")
  out = UNITS / f"{name}.json"
  if out.exists():
    return out.name, "skip"
  t0 = time.time()
  env = state(pol, rep, t)
  info = perturb(env, *pert) if pert else {}
  res = {}
  if info is not None:
    res = {",".join(map(str, d)): value(env, d) for d in drops}
  UNITS.mkdir(parents=True, exist_ok=True)
  feat = None
  if pert and info is not None:
    own, ctx = feats(env, env.o.sid.index(pert[0]))
    feat = dict(own=own, ctx=ctx)
  out.write_text(json.dumps(dict(pol=pol, rep=rep, t=t, pert=pert, info=info, values=res, feat=feat, sec=time.time() - t0)), encoding="utf-8")
  return out.name, f"{time.time() - t0:.0f}s"


# 계획: 상태별 공급자 feature, 섭동 대상 (Myopic·Rule에서 같은 수, 무작위)
def plan():
  P, g = [], np.random.default_rng([CFG.seed, 11])
  for pol in POLS:
    for rep in REPS:
      for t in TS:
        env = state(pol, rep, t)
        sup = []
        for k, s in enumerate(env.o.sid):
          own, ctx = feats(env, k)
          sup.append(dict(sid=int(s), own=own, ctx=ctx))
        P.append(dict(pol=pol, rep=rep, t=t, sup=sup, pert=[]))
  for pol in POLS:
    idx = [i for i, s in enumerate(P) if s["pol"] == pol]
    for i in g.choice(idx, N_PERT // 2, replace=False):
      P[int(i)]["pert"].append(int(g.choice([s["sid"] for s in P[int(i)]["sup"]])))
  PLAN.write_text(json.dumps(P), encoding="utf-8")
  return P


if __name__ == "__main__":
  if "smoke" in sys.argv:
    env = state("Myopic", 0, 6)
    print(feats(env, 0))
    for case in CASES:
      e = copy.deepcopy(env)
      print(case, perturb(e, env.o.sid[0], case), len(e.o.sid), e.o.wait.tolist())
  if "run" in sys.argv:
    P = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else plan()
    jobs = []
    for s in P:
      sids = [x["sid"] for x in s["sup"]]
      drops = [[]] + [[i] for i in sids]
      jobs += [(s["pol"], s["rep"], s["t"], None, c, drops[i:i + 7]) for c, i in enumerate(range(0, len(drops), 7))]
      for sid in s["pert"]:
        jobs += [(s["pol"], s["rep"], s["t"], (sid, c), 0, [[], [sid]]) for c in CASES]
    print(f"상태 {len(P)}, 공급자 {sum(len(s['sup']) for s in P)}, 섭동 대상 {sum(len(s['pert']) for s in P)}, 작업 {len(jobs)}", flush=True)
    jobs.sort(key=lambda j: j[3] is None)
    t0 = time.time()
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} (경과 {time.time() - t0:.0f}s)", flush=True)
