"""슬라이스 3 진단·보정: 새 규모에서 환경이 연구 질문을 표현하는지 확인하고 capacity와 반응 기준 잉여율을 다시 잡는다.
Cfg 기본값은 바꾸지 않고 replace로 후보 환경(품목 15, 공급자 자리 24, n_alt 4, 묶음 2, 편중도 0.5)을 만든다.
occ → capacity → 마진 → 잉여율 → 기울기 → 재참여 → occ의 순환이라 measure를 고정점까지 반복하고 경로를 남긴다.
"""
import json
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg
from utils.instance import qty_range
from env.platform import Env, worth
from match.milp_solve import Policy
from bench.tune import ref_rate, occupancy, slopes
from result import diag

OUT = diag.RUNS / "diag_s3_scaleup.json"
BASE = dict(n_items=15, n_sup=24, n_alt=4, items_per_order=2, conc=0.5)
REPS = 10
ITERS = 6
TOL = 0.01  # 고정점 판정: r_ref와 occ의 상대 변화


# 후보 환경 (Cfg 기본값은 그대로 두고 replace로만 만든다)
def cand(**kw):
  return replace(Cfg(), **{**BASE, **kw})


# 품목별 안정 상태 수요와 총 공급용량 (instance.make와 같은 식)
def cap_of(cfg):
  lo, hi = qty_range(cfg)
  dem = cfg.lam_buy / (1 - cfg.ret_ss) * cfg.items_per_order / cfg.n_items * (lo + hi) / 2
  return dem, round(cfg.cover * dem / cfg.occ)


# 규칙 기반 sh_ref로 돌리며 기간별 진단: 점유율, 공급 부족, 충족률, 배정 집중도
def probe(cfg, reps=REPS):
  tc = replace(cfg, seed=cfg.tune_seed)
  dem, tot = cap_of(cfg)
  half = tc.T // 2
  sup, nocov, short, fill, hhi, top, degs, maxcap, solo = [], [], [], [], [], [], [], [], []
  for rep in range(reps):
    env, pol = Env(tc, rep), Policy(tc, split=tc.sh_ref)
    env.reset()
    degs += env.inst.sup_items.sum(1).tolist()
    maxcap.append(float(env.inst.sup_cap.max()))
    for t in range(tc.T):
      o = env.o
      x, _, _ = pol.act(o)
      if t >= half:
        sup.append(len(o.sid))
        act = o.cap.sum(0) if len(o.cap) else np.zeros(tc.n_items)
        need = o.q.sum(0) if len(o.q) else np.zeros(tc.n_items)
        nocov.append(int((act == 0).sum()))
        short.append(float(np.mean(act < need)))
        solo.append(float(np.mean(o.cap.max(0) >= need)) if len(o.cap) else 0.0)
        # 배정 집중도: 실제로 배정된 품목마다 최대 공급자 몫과 허핀달 지수
        v = x.sum(0)  # 공급자 × 품목
        for i in range(tc.n_items):
          s = v[:, i].sum()
          if s > 0:
            sh = v[:, i] / s
            top.append(float(sh.max()))
            hhi.append(float((sh ** 2).sum()))
      r = env.step(pol.act(o))
      if t >= half:
        fill.append(r["n_ok"] / max(r["n_buy"], 1))
  m = lambda a: float(np.mean(a)) if len(a) else 0.0
  return dict(dem=dem, cap_item=tot, cap_sup=tot / cfg.n_alt, cap_sup_max=m(maxcap),
              one_covers_by_design=bool(tot / cfg.n_alt >= dem), one_covers_observed=m(solo),
              occ=m(sup) / cfg.n_sup, sup=m(sup), sup_min=int(np.min(sup)), sup_max=int(np.max(sup)),
              nocov=m(nocov), nocov_max=int(np.max(nocov)), short=m(short), fill=m(fill),
              assign_top=m(top), assign_hhi=m(hhi), deg_mean=m(degs), deg_max=int(np.max(degs)),
              deg_var=float(np.var(degs)))


# 고정점 보정: 기준 잉여율 → 기울기 → 점유율 → capacity → 다시 기준 잉여율
def calibrate(cfg, iters=ITERS):
  path = []
  for it in range(iters):
    rb, rs = ref_rate(cfg)
    nxt = slopes(cfg, rb, rs)
    oc = occupancy(nxt)
    nxt = replace(nxt, occ=oc)
    dem, tot = cap_of(nxt)
    d = max(abs(rb - cfg.r_ref_buy) / cfg.r_ref_buy, abs(rs - cfg.r_ref_sup) / cfg.r_ref_sup,
            abs(oc - cfg.occ) / cfg.occ)
    path.append(dict(it=it, r_ref_buy=rb, r_ref_sup=rs, b_buy=nxt.b_buy, b_sup=nxt.b_sup,
                     occ=oc, cap_item=tot, rel_change=float(d)))
    cfg = nxt
    if d <= TOL:
      break
  return cfg, path


# 편중도에 따라 지표가 달라지는지 (구조가 결과를 바꿀 여지가 있는지)
def conc_sweep(cfg, concs=(0.0, 0.5, 1.0)):
  return {f"conc{c:g}": probe(replace(cfg, conc=c), reps=6) for c in concs}


# 연구 질문 표현 적절성 판정 (기준은 결과를 보기 전에 정한다)
def verdict(p):
  return {
    "배정이 실제 선택 (한 공급자가 품목 수요를 단독 충족하지 못함)": bool(not p["one_covers_by_design"] and p["one_covers_observed"] < 0.5),
    "시장이 유지됨 (점유율 0.2~0.95)": bool(0.2 < p["occ"] < 0.95),
    "묶음이 깨지는 일이 있고 시장이 죽지도 않음 (충족률 0.3~0.8)": bool(0.3 <= p["fill"] <= 0.8),
    "공급자를 잃는 손해가 실재 (공급자 없는 품목 > 0)": bool(p["nocov"] > 0),
    "공급 부족이 과하지 않음 (공급자 없는 품목 평균 < 품목 수의 10%)": bool(p["nocov"] < 0.1 * p["n_items"]),
    "배정이 한 공급자에 몰리지 않음 (품목별 최대 몫 < 0.9)": bool(p["assign_top"] < 0.9),
  }


NOTE = ("슬라이스 3 진단·보정. Cfg 기본값은 바꾸지 않고 replace로 후보 환경을 만든다 "
        f"(기본 후보 {BASE}). occ → capacity → 마진 → 잉여율 → 기울기 → 재참여 → occ의 순환이라 "
        f"measure를 상대 변화 {TOL} 이하까지 반복한다 (재참여율을 맞추는 이분탐색이 아니라 측정의 자기일관성 확보). "
        f"진단은 튜닝 seed {REPS}반복의 후반 절반 기간, 편중도 sweep은 6반복이다.")

if __name__ == "__main__":
  t0 = time.time()
  c0 = cand()
  res = dict(note=NOTE, base=BASE, cfg_default=asdict(Cfg()))
  print("A 보정 전 진단", flush=True)
  before = probe(c0)
  before["n_items"] = c0.n_items
  res["before"] = dict(cfg={k: getattr(c0, k) for k in ("occ", "r_ref_buy", "r_ref_sup", "b_buy", "b_sup")}, probe=before,
                       verdict=verdict(before))
  print("B 고정점 보정", flush=True)
  c1, path = calibrate(c0)
  res["calibration"] = dict(path=path, converged=bool(path[-1]["rel_change"] <= TOL),
                            final={k: getattr(c1, k) for k in ("occ", "r_ref_buy", "r_ref_sup", "b_buy", "b_sup")})
  print("C 보정 후 진단", flush=True)
  after = probe(c1)
  after["n_items"] = c1.n_items
  res["after"] = dict(probe=after, verdict=verdict(after))
  print("D 편중도 sweep", flush=True)
  res["conc_sweep"] = conc_sweep(c1)
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
