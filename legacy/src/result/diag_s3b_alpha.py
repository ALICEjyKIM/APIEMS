"""슬라이스 3b: 공급자 용량 편중을 줄이고 다시 보정한다.
품목 용량을 대체 공급자에게 나누는 Dirichlet 집중 파라미터 dir_alpha만 올린다 (총 공급량과 cover는 그대로).
후보 값을 훑어 단독 충족 비율과 배정 집중도가 얼마나 줄는지 보고, 고른 값에서 occ·기준 잉여율을 재보정해 6개 기준을 다시 확인한다.
"""
import json
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg, SCALE15
from utils.instance import make
from result.diag_s3_scaleup import probe, calibrate, conc_sweep, verdict, cap_of
from result import diag

OUT = diag.RUNS / "diag_s3b_alpha.json"
ALPHAS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
SWEEP_REPS = 8
# 고른 값: 훑기에서 단독 충족 비율과 배정 최대 몫의 개선이 평탄해지는 지점.
# 1 → 3에서 단독 충족이 0.175 → 0.095로 줄고 3 → 5에서 0.083까지 내려가지만 5 → 10 → 20은 0.083/0.089/0.071로 잡음 수준이다.
# 20까지 올리면 용량 변동계수가 0.271로 떨어져 공급자 간 규모 차이가 거의 사라지므로, 꼬리만 자르고 이질성은 남기는 5를 쓴다.
PICK = 5.0


# SCALE15 환경 (보정값 포함)
def base(**kw):
  return replace(Cfg(), **{**SCALE15, **kw})


# dir_alpha만 바꿔가며 훑는다 (보정은 하지 않아 비교가 같은 조건에서 된다)
def sweep(alphas=ALPHAS):
  out = []
  for a in alphas:
    cfg = base(dir_alpha=a)
    caps = np.concatenate([make(cfg, r).sup_cap.ravel() for r in range(SWEEP_REPS)])
    pos = caps[caps > 0]
    p = probe(cfg, reps=SWEEP_REPS)
    p["n_items"] = cfg.n_items
    out.append(dict(dir_alpha=a, cap_p50=float(np.percentile(pos, 50)), cap_p95=float(np.percentile(pos, 95)),
                    cap_max=float(pos.max()), cap_cv=float(pos.std() / pos.mean()), probe=p, verdict=verdict(p)))
  return out


NOTE = ("슬라이스 3b: dir_alpha만 올려 공급자 용량 편중을 줄인다. 총 공급량(tot)과 cover는 바꾸지 않는다 "
        f"(Dirichlet 집중만 바꾸므로 합은 항상 tot다). 훑기는 SCALE15 보정값을 고정한 채 {SWEEP_REPS}반복이고, "
        "고른 값에서 occ·기준 잉여율을 고정점까지 다시 재고 6개 기준과 편중도 sweep을 다시 확인한다.")

if __name__ == "__main__":
  t0 = time.time()
  res = dict(note=NOTE, alphas=list(ALPHAS), scale15=SCALE15)
  print("A dir_alpha 훑기", flush=True)
  res["sweep"] = sweep()
  for r in res["sweep"]:
    p = r["probe"]
    print(f"  alpha {r['dir_alpha']:5.1f} | 용량 중앙 {r['cap_p50']:5.1f} 95% {r['cap_p95']:5.1f} 최대 {r['cap_max']:5.1f} 변동계수 {r['cap_cv']:.3f} "
          f"| 단독충족 {p['one_covers_observed']:.3f} 최대몫 {p['assign_top']:.3f} 허핀달 {p['assign_hhi']:.3f} "
          f"충족률 {p['fill']:.3f} 점유율 {p['occ']:.3f} 공급자없는품목 {p['nocov']:.2f}", flush=True)
  pick = PICK
  res["pick"] = pick
  print(f"B 고른 dir_alpha = {pick}, 고정점 보정", flush=True)
  c1, path = calibrate(base(dir_alpha=pick))
  res["calibration"] = dict(path=path, converged=bool(path[-1]["rel_change"] <= 0.01),
                            final={k: getattr(c1, k) for k in ("occ", "r_ref_buy", "r_ref_sup", "b_buy", "b_sup")})
  print("C 보정 후 진단", flush=True)
  after = probe(c1)
  after["n_items"] = c1.n_items
  dem, tot = cap_of(c1)
  res["after"] = dict(cfg={k: getattr(c1, k) for k in ("dir_alpha", "occ", "r_ref_buy", "r_ref_sup", "b_buy", "b_sup")},
                      dem=dem, cap_item=tot, probe=after, verdict=verdict(after))
  print("D 편중도 sweep", flush=True)
  res["conc_sweep"] = conc_sweep(c1)
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
