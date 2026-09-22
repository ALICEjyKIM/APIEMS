"""슬라이스 3 확인 실험 (체크포인트).
튜닝 seed에서 선형 가치 근사를 학습하고, 평가 seed에서 선형 유지 가치 정책을 규칙 기반(주 비교)·근시안(참고)과 비교한다.
검증 반복의 표본 상태에서 시뮬레이션 기준치 대비 예측 오차와 유지 가치 변환 오차를 잰다. 원자료는 result/runs/diag_s3_*.json.
"""
import json
import sys
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg
from env.platform import Env
from match.milp_solve import Policy
from bench.rule_split import rule
from vfa.train import collect, val_mask
from vfa.linear import select
from vfa.coef import raw, mc_coefs, conv_err, vfa_policy
from utils.features import phi
from match.mc_value import mc_value
from result.diag import run, table, RUNS

MC_T = (5, 10, 15, 20)  # 예측 오차를 잴 기간
CONV_N = 4  # 변환 오차를 잴 상태 수 (참여자마다 기준치를 다시 돌려 비용이 커서 앞쪽 상태만)


# 학습: 튜닝 seed 데이터로 λ를 고르고 맞춘 모델과 적합 기록
def fit(cfg):
  X, y, g = collect(cfg)
  m, mse = select(cfg, X, y, g)
  va = val_mask(cfg, g)
  rec = dict(n=len(y), n_val=int(va.sum()), lam=m.lam, val_mse={str(k): v for k, v in mse.items()},
             val_r2=float(1 - mse[m.lam] / np.var(y[va])), y_mean=float(y.mean()), y_sd=float(y.std()), w=m.w.tolist(), b=m.b)
  return m, rec


# 검증 반복(튜닝 seed)의 표본 상태에서 모델 예측 vs 시뮬레이션 기준치, 앞쪽 CONV_N개 상태는 유지 가치 변환 오차도
def mc_check(cfg, m):
  tc = replace(cfg, seed=cfg.tune_seed)
  out = []
  for rep in range(round(cfg.vf_reps * (1 - cfg.vf_val)), cfg.vf_reps):
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    for t in range(max(MC_T) + 1):
      if t in MC_T:
        cb, cs, v = raw(m, env.o)
        row = dict(rep=rep, t=t, pred=float(v), mc=mc_value(env).tolist())
        if len(out) < CONV_N:
          mb, ms, _ = mc_coefs(env)
          row.update(c_buy=cb.tolist(), c_sup=cs.tolist(), mc_buy=mb.tolist(), mc_sup=ms.tolist())
        out.append(row)
      env.step(pol.act(env.o))
  return out


# 예측·변환 오차 요약 (json의 원자료로만)
def mc_table(path):
  d = json.loads(path.read_text(encoding="utf-8"))
  rows = d["states"]
  e = np.array([r["pred"] - np.mean(r["mc"]) for r in rows])
  se = np.array([np.std(r["mc"], ddof=1) / np.sqrt(len(r["mc"])) for r in rows])
  mc = np.array([np.mean(r["mc"]) for r in rows])
  cv = [r for r in rows if "c_buy" in r]
  cb = conv_err(np.concatenate([r["c_buy"] for r in cv]), np.concatenate([r["mc_buy"] for r in cv]))
  cs = conv_err(np.concatenate([r["c_sup"] for r in cv]), np.concatenate([r["mc_sup"] for r in cv]))
  mcb, mcs = np.mean(np.concatenate([r["mc_buy"] for r in cv])), np.mean(np.concatenate([r["mc_sup"] for r in cv]))
  mdb, mds = np.mean(np.concatenate([r["c_buy"] for r in cv])), np.mean(np.concatenate([r["c_sup"] for r in cv]))
  return "\n".join([
    f"원자료: {path.name}. {d['note']}", "",
    "| 항목 | 값 |", "|---|---|",
    f"| 학습 표본 수 / 검증 표본 수 | {d['fit']['n']} / {d['fit']['n_val']} |",
    f"| 고른 λ (검증 MSE: {', '.join(f'{k}: {v:.3g}' for k, v in d['fit']['val_mse'].items())}) | {d['fit']['lam']} |",
    f"| 검증 R² (목표 평균 {d['fit']['y_mean']:.0f}, 표준편차 {d['fit']['y_sd']:.0f}) | {d['fit']['val_r2']:.3f} |",
    f"| 예측 오차 (예측 − 기준치) 평균 ± 95% CI, 상태 {len(rows)}개 | {e.mean():+.0f} ± {1.96 * e.std(ddof=1) / np.sqrt(len(e)):.0f} |",
    f"| 예측 절대 오차 평균 / 기준치 평균 | {np.abs(e).mean():.0f} / {mc.mean():.0f} |",
    f"| 기준치 자체의 표준오차 평균 (rollout {len(rows[0]['mc'])}개) | {se.mean():.0f} |",
    f"| 변환 오차 주문자 (상태 {len(cv)}개): 평균 절대 차이 / 모델 c 평균 / 기준치 c 평균 | {cb:.0f} / {mdb:.0f} / {mcb:.0f} |",
    f"| 변환 오차 공급자 (상태 {len(cv)}개): 평균 절대 차이 / 모델 c 평균 / 기준치 c 평균 | {cs:.0f} / {mds:.0f} / {mcs:.0f} |",
  ])


if __name__ == "__main__":
  cfg, t0 = Cfg(), time.time()
  m, rec = fit(cfg)
  print(f"fit {time.time() - t0:.0f}s: lam={rec['lam']} val_r2={rec['val_r2']:.3f}", flush=True)
  if "eval" in sys.argv:
    pol = {"근시안": lambda c: Policy(c), "규칙 기반 (0.3, 0.3)": lambda c: rule(c), "선형 유지 가치": lambda c: vfa_policy(c, m)}
    path = run(cfg, pol, "s3_eval", f"슬라이스 3: 선형 유지 가치(λ = {rec['lam']}, 튜닝 seed 학습) vs 규칙 기반 (0.3, 0.3)·근시안, 평가 seed 30반복")
    print(table(path, "규칙 기반 (0.3, 0.3)", "선형 유지 가치"), "\n")
    print(table(path, "근시안", "선형 유지 가치", "규칙 기반 (0.3, 0.3)"), "\n", flush=True)
  if "mc" in sys.argv:
    path = RUNS / "diag_s3_mc.json"
    path.write_text(json.dumps(dict(note=f"선형 가치 모델 vs 시뮬레이션 기준치 (튜닝 seed 검증 반복, 기간 {MC_T}, rollout {cfg.mc_R}개, H {cfg.vf_H})",
                                    cfg=asdict(cfg), fit=rec, states=mc_check(cfg, m))), encoding="utf-8")
    print(mc_table(path))
  print(f"total {time.time() - t0:.0f}s")
