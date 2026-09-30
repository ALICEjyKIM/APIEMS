"""슬라이스 4b: 학습 타깃의 미래 rollout 수 M을 정한다.
baseline 칸 하나에서 학습 타깃을 M = 1, 2, 5의 Monte Carlo 평균으로 만들고, 고정된 M = 10 검증 타깃으로 평가한다.
미래를 앞쪽부터 겹쳐 쓰므로 세 M이 같은 표본에서 나오며(짝지은 비교), 계산비용은 M에 비례한다.
"""
import json
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg, SCALE15
from utils.features import phi
from vfa.train import collect_futures, target, val_mask
from vfa import linear, mlp
from result import diag

OUT = diag.RUNS / "diag_s4b_train_m.json"
MS = (1, 2, 5)
K_TRAIN, K_VAL = max(MS), 10


# baseline 칸 (확정 환경, 학습 반복 vf_reps3)
def cell_cfg():
  c = Cfg()
  return replace(c, **SCALE15, vf_reps=c.vf_reps3)


# 외부 분할로 격자를 골라 검증 성능을 낸다 (학습 분할에만 맞추고 전체 재적합은 하지 않는다: M별 타깃이 섞이지 않게)
def fit_grid(cfg, make, grid, Xtr, ytr, Xva, yva):
  mse = {}
  for a in grid:
    m = make(a).fit(Xtr, ytr)
    mse[a] = float(np.mean((m.predict(Xva) - yva) ** 2))
  a = min(mse, key=mse.get)
  vv = float(np.var(yva))
  return dict(grid=[float(x) for x in grid], val_mse=[mse[x] for x in grid], pick=float(a),
              edge=bool(a in (grid[0], grid[-1])), val_r2=float(1 - mse[a] / vv),
              train_r2=float(1 - np.mean((make(a).fit(Xtr, ytr).predict(Xtr) - ytr) ** 2) / np.var(ytr)))


NOTE = ("슬라이스 4b: 학습 타깃 M 결정. 확정 환경(SCALE15)의 baseline 칸에서 학습 타깃을 M = 1, 2, 5의 "
        "Monte Carlo 평균으로 만들고 고정된 M = 10 검증 타깃으로 평가한다. 미래를 앞쪽부터 겹쳐 쓰므로 "
        "세 M이 같은 표본에서 나온다. 격자는 학습 분할에만 맞추고 전체 재적합은 하지 않는다 (M별 타깃이 섞이지 않게). "
        "Linear·MLP를 함께 낸 것은 M 판단이 모델에 흔들리지 않는지 보기 위한 것이며, 근사 방법 비교가 아니다 "
        "(그 비교는 같은 튜닝 예산으로 뒤에 한다).")

if __name__ == "__main__":
  t0, cfg = time.time(), cell_cfg()
  r0 = round(cfg.vf_reps * (1 - cfg.vf_val))
  print(f"학습 반복 0~{r0 - 1}, 검증 반복 {r0}~{cfg.vf_reps - 1}", flush=True)
  t = time.time()
  Otr, Vtr, gtr = collect_futures(cfg, K_TRAIN, range(r0))
  sec_tr = time.time() - t
  print(f"A 학습 상태 {len(Otr)}개 × 미래 {K_TRAIN}개  {sec_tr:.0f}s", flush=True)
  t = time.time()
  Ova, Vva, gva = collect_futures(cfg, K_VAL, range(r0, cfg.vf_reps))
  sec_va = time.time() - t
  print(f"B 검증 상태 {len(Ova)}개 × 미래 {K_VAL}개  {sec_va:.0f}s", flush=True)
  Xtr, Xva = np.array([phi(o) for o in Otr]), np.array([phi(o) for o in Ova])
  yva = target(Vva, K_VAL)
  noise = float(Vva.var(1, ddof=1).mean())
  res = dict(note=NOTE, cfg=asdict(cfg), n_train=len(Otr), n_val=len(Ova), K_train=K_TRAIN, K_val=K_VAL,
             sec_collect_train=sec_tr, sec_collect_val=sec_va,
             sec_per_future=(sec_tr / (len(Otr) * K_TRAIN)),
             val_noise_var=noise, val_target_var=float(np.var(yva)),
             val_ceiling=float(1 - (noise / K_VAL) / np.var(yva)),
             Vtr=Vtr.tolist(), Vva=Vva.tolist(), rows=[])
  for m in MS:
    ytr = target(Vtr, m)
    row = dict(M=m, train_target_sd=float(ytr.std()), train_noise_se=float(np.sqrt(Vtr.var(1, ddof=1).mean() / m)),
               solves=int(len(Otr) * m * cfg.vf_H), sec=float(len(Otr) * m * res["sec_per_future"]))
    for name, sel, grid in (("선형", lambda a: linear.Linear(a), cfg.lin_lams),
                            ("MLP", lambda a: mlp.MLP(cfg, a), cfg.nn_wds)):
      row[name] = fit_grid(cfg, sel, grid, Xtr, ytr, Xva, yva)
      print(f"  M={m} {name}: val R² {row[name]['val_r2']:.4f} (고른 값 {row[name]['pick']}, 격자 끝 {row[name]['edge']})", flush=True)
    res["rows"].append(row)
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
