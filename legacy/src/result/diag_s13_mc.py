"""진단 s13 MC: (1) 상태 확장: diag s11과 같은 환경·정책·기간 규칙·c_i 계산으로 반복 8~15의 상태 48개를 더 잰다.
(2) 연결 재배치(swap): s11 섭동 대상 24개에서 공급자 i의 품목 a를 함께 원하는 두 주문자가 묶음의 다른 품목(과 그 수량)을 맞바꾼다.
품목별 요구량과 i의 context 요약은 그대로이고 누가 어떤 품목을 함께 원하는지만 바뀐다. c_i는 s11과 같은 키로 다시 잰다 (Rule-continuation).
"""
import json
import sys
import time
from multiprocessing import Pool
import numpy as np
from result.diag_s11_context import CFG, POLS, TS, state, value, feats, PLAN as PLAN11
from result import diag

UNITS = diag.RUNS / "diag_s13_units"
PLAN = diag.RUNS / "diag_s13_plan.json"
REPS = range(8, 16)
WORKERS = 12


# 재배치: i의 품목 a를 원하는 주문자 b1, b2 (묶음의 다른 품목 x, y, x ≠ y, b1은 y를 b2는 x를 원하지 않음) 중
# 활동 공급자 수 차이 |deg x − deg y|가 가장 큰 짝을 골라 x와 y를 수량과 함께 맞바꾼다 (가격은 각 주문자의 마크업 유지). 없으면 None
def swap(env, sid):
  o = env.o
  k = o.sid.index(sid)
  it = o.cap[k] > 0
  deg = (o.cap > 0).sum(0)
  best = None
  bids = list(env.ords)
  for a in np.where(it)[0]:
    W = [b for b in bids if env.ords[b].qty[a] > 0]
    for i1 in range(len(W)):
      for i2 in range(i1 + 1, len(W)):
        p1, p2 = env.ords[W[i1]], env.ords[W[i2]]
        for x in np.where((p1.qty > 0) & ~it)[0]:
          for y in np.where((p2.qty > 0) & ~it)[0]:
            if x == y or p1.qty[y] > 0 or p2.qty[x] > 0:
              continue
            gap = abs(int(deg[x]) - int(deg[y]))
            if best is None or gap > best[0]:
              best = (gap, W[i1], W[i2], int(x), int(y), int(a))
  if best is None or best[0] == 0:
    return None
  gap, b1, b2, x, y, a = best
  base = env.inst.base
  pairs = [(env.ords[b1], env.ords[b2])]
  if env.buys[b1] is not env.ords[b1] or env.buys[b2] is not env.ords[b2]:
    pairs.append((env.buys[b1], env.buys[b2]))
  for P1, P2 in pairs:
    q1x, q2y, m1, m2 = int(P1.qty[x]), int(P2.qty[y]), P1.price[x] / base[x], P2.price[y] / base[y]
    for P, old, new, q, mk in ((P1, x, y, q2y, m1), (P2, y, x, q1x, m2)):
      P.qty[new], P.price[new], P.items[new] = q, base[new] * mk, True
      P.qty[old], P.price[old], P.items[old] = 0, 0.0, False
  env._observe()
  return dict(b1=int(b1), b2=int(b2), x=x, y=y, a=a, deg_gap=gap)


# 작업 단위: ("state", 정책, 반복, 기간, 묶음, 제거 집합들) 또는 ("swap", 정책, 반복, 기간, 공급자)
def unit(job):
  kind = job[0]
  t0 = time.time()
  if kind == "state":
    _, pol, rep, t, c, drops = job
    out = UNITS / f"{pol}_r{rep}_t{t}_c{c}.json"
    if out.exists():
      return out.name, "skip"
    env = state(pol, rep, t)
    res = dict(pol=pol, rep=rep, t=t, values={",".join(map(str, d)): value(env, d) for d in drops})
  else:
    _, pol, rep, t, sid = job
    out = UNITS / f"{pol}_r{rep}_t{t}_swap_s{sid}.json"
    if out.exists():
      return out.name, "skip"
    env = state(pol, rep, t)
    before = feats(env, env.o.sid.index(sid))
    info = swap(env, sid)
    res = dict(pol=pol, rep=rep, t=t, sid=sid, info=info, values={}, feat_before=before)
    if info is not None:
      res["feat_after"] = feats(env, env.o.sid.index(sid))
      res["values"] = {"": value(env, []), str(sid): value(env, [sid])}
  res["sec"] = time.time() - t0
  UNITS.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res), encoding="utf-8")
  return out.name, f"{res['sec']:.0f}s"


# 확장 상태 계획 (s11과 같은 규칙)
def plan():
  P = []
  for pol in POLS:
    for rep in REPS:
      for t in TS:
        env = state(pol, rep, t)
        sup = []
        for k, s in enumerate(env.o.sid):
          own, ctx = feats(env, k)
          sup.append(dict(sid=int(s), own=own, ctx=ctx))
        P.append(dict(pol=pol, rep=rep, t=t, sup=sup, pert=[]))
  PLAN.write_text(json.dumps(P), encoding="utf-8")
  return P


if __name__ == "__main__":
  if "smoke" in sys.argv:
    P11 = json.loads(PLAN11.read_text(encoding="utf-8"))
    for s in [s for s in P11 if s["pert"]][:6]:
      env = state(s["pol"], s["rep"], s["t"])
      sid = s["pert"][0]
      b = feats(env, env.o.sid.index(sid))
      info = swap(env, sid)
      if info:
        a = feats(env, env.o.sid.index(sid))
        print(info, "own 같음", a[0] == b[0], "ctx 같음", a[1] == b[1], {k: (b[1][k], a[1][k]) for k in a[1] if a[1][k] != b[1][k]})
      else:
        print("적용 불가", s["pol"], s["rep"], s["t"], sid)
  if "run" in sys.argv:
    P = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else plan()
    P11 = json.loads(PLAN11.read_text(encoding="utf-8"))
    jobs = [("swap", s["pol"], s["rep"], s["t"], s["pert"][0]) for s in P11 if s["pert"]]
    for s in P:
      drops = [[]] + [[x["sid"]] for x in s["sup"]]
      jobs += [("state", s["pol"], s["rep"], s["t"], c, drops[i:i + 7]) for c, i in enumerate(range(0, len(drops), 7))]
    print(f"확장 상태 {len(P)}, 공급자 {sum(len(s['sup']) for s in P)}, 작업 {len(jobs)}", flush=True)
    t0 = time.time()
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} (경과 {time.time() - t0:.0f}s)", flush=True)
