"""실험 1: 선형 vs MLP (같은 시장 요약 지표, 같은 학습 데이터·검증 분할·튜닝 예산).
value 단계(튜닝 seed만): 두 모델을 학습하고 시뮬레이션 기준치 대비 예측 오차와 유지 가치 변환 오차를 재서 result/runs/exp1_value.json에 저장한다.
eval 단계(평가 seed, 한 번): 근시안·규칙 기반·선형·MLP 유지 가치 정책을 비교해 result/runs/exp1_eval.json에 저장한다. 표는 json에서만 만든다.
"""
import json
import sys
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg
from env.platform import Env
from match.milp_solve import Policy
from match.mc_value import mc_value
from bench.rule_split import rule
from vfa.train import collect, val_mask
from vfa import linear, mlp
from vfa.coef import raw, mc_coefs, conv_err, vfa_policy
from result import diag

PRED_T = (5, 10, 15, 20)  # 예측 오차 상태의 기간 (슬라이스 3과 같음)
CONV_T = (10, 20)  # 변환 오차 상태의 기간 (10기간 간격이라 conv_R개 미래 키가 다른 상태와 겹치지 않음)
NAMES = ("선형", "MLP")
PAIRS = [("MLP 유지 가치", "근시안"), ("MLP 유지 가치", "선형 유지 가치"), ("선형 유지 가치", "근시안"), ("규칙 기반 (0.3, 0.3)", "근시안")]


# 학습: 같은 데이터·분할로 선형(λ)과 MLP(weight decay)를 각자 격자에서 고른다 → (이름 → 모델, 적합 기록)
def models(cfg):
  X, y, g = collect(cfg)
  va, out, rec = val_mask(cfg, g), {}, dict(n=len(y), n_val=int(val_mask(cfg, g).sum()), y_mean=float(y.mean()), y_sd=float(y.std()))
  for n, sel, grid, key in (("선형", linear.select, cfg.lin_lams, "lam"), ("MLP", mlp.select, cfg.nn_wds, "wd")):
    m, mse = sel(cfg, X, y, g)
    a = getattr(m, key)
    out[n] = m
    rec[n] = dict(grid=list(grid), val_mse=[mse[k] for k in grid], pick=a, edge=a in (grid[0], grid[-1]), val_r2=float(1 - mse[a] / np.var(y[va])))
  return out, rec


# 튜닝 seed 검증 반복을 규칙 기반으로 돌리며 기간 ts의 (반복, 기간, 환경)을 낸다 (환경은 다음 기간으로 넘어가기 전에만 쓴다)
def states(cfg, ts):
  tc = replace(cfg, seed=cfg.tune_seed)
  for rep in range(round(cfg.vf_reps * (1 - cfg.vf_val)), cfg.vf_reps):
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    for t in range(max(ts) + 1):
      if t in ts:
        yield rep, t, env
      env.step(pol.act(env.o))


# value 단계: 적합 기록, 상태별 예측(두 모델)·기준치, 상태별 모델 c(가드 전)·기준치 c·그 표준오차를 저장한다
def value(cfg, pred_t=PRED_T, conv_t=CONV_T):
  ms, fit = models(cfg)
  pred = [dict(rep=rep, t=t, mc=mc_value(env).tolist(), **{n: float(raw(ms[n], env.o)[2]) for n in NAMES}) for rep, t, env in states(cfg, pred_t)]
  conv = []
  for rep, t, env in states(cfg, conv_t):
    cb, cs, _, sb, ss = mc_coefs(env, cfg.conv_R)
    row = dict(rep=rep, t=t, mc_buy=cb.tolist(), mc_sup=cs.tolist(), se_buy=sb.tolist(), se_sup=ss.tolist())
    for n in NAMES:
      c = raw(ms[n], env.o)
      row[n] = dict(buy=c[0].tolist(), sup=c[1].tolist())
    conv.append(row)
  note = f"실험 1 value (튜닝 seed 검증 반복): 예측 오차 기간 {pred_t}·rollout {cfg.mc_R}, 변환 오차 기간 {conv_t}·rollout {cfg.conv_R}, H {cfg.vf_H}"
  path = diag.RUNS / "exp1_value.json"
  path.write_text(json.dumps(dict(note=note, cfg=asdict(cfg), fit=fit, pred=pred, conv=conv)), encoding="utf-8")
  return path


# eval 단계 (평가 seed, 한 번): 다시 맞춘 모델이 value 단계 적합과 같은지 확인하고 네 정책을 비교·요약한다
def evaluate(cfg):
  ms, fit = models(cfg)
  saved = json.loads((diag.RUNS / "exp1_value.json").read_text(encoding="utf-8"))["fit"]
  assert all(fit[n]["pick"] == saved[n]["pick"] and np.allclose(fit[n]["val_mse"], saved[n]["val_mse"]) for n in NAMES)
  pol = {"근시안": lambda c: Policy(c), "규칙 기반 (0.3, 0.3)": lambda c: rule(c),
         "선형 유지 가치": lambda c: vfa_policy(c, ms["선형"]), "MLP 유지 가치": lambda c: vfa_policy(c, ms["MLP"])}
  note = f"실험 1 eval (평가 seed {cfg.seed}, {cfg.reps}반복, 한 번): 선형 λ = {fit['선형']['pick']}, MLP weight decay = {fit['MLP']['pick']}"
  path = diag.run(cfg, pol, "eval", note, prefix="exp1")
  diag.summarize(path, PAIRS)
  return path


# value json의 표: 적합(설정별 검증 MSE, 고른 설정, 격자 끝), 예측 오차(편향, 절대 오차, 선형 − MLP), 변환 오차(모델 c vs 기준치 c, 잡음 바닥)
def value_table(path):
  d = json.loads(path.read_text(encoding="utf-8"))
  f, P, C = d["fit"], d["pred"], d["conv"]
  mc = np.array([np.mean(r["mc"]) for r in P])
  e = {n: np.array([r[n] for r in P]) - mc for n in NAMES}
  ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
  out = [f"원자료: {path.name}. {d['note']}", "", f"학습 표본 {f['n']}개 (검증 {f['n_val']}개), 목표 평균 {f['y_mean']:.0f}, 표준편차 {f['y_sd']:.0f}", "",
         "| 모델 | 격자 (검증 MSE) | 고른 설정 | 격자 끝 | 검증 R² |", "|---|---|---|---|---|"]
  for n in NAMES:
    out.append(f"| {n} | {', '.join(f'{a}: {m:.4g}' for a, m in zip(f[n]['grid'], f[n]['val_mse']))} | {f[n]['pick']} | {'예' if f[n]['edge'] else '아니오'} | {f[n]['val_r2']:.3f} |")
  se = np.mean([np.std(r["mc"], ddof=1) / np.sqrt(len(r["mc"])) for r in P])
  out += ["", f"예측 오차 (상태 {len(P)}개, 기준치 평균 {mc.mean():.0f}, 기준치 표준오차 평균 {se:.0f})", "",
          "| 모델 | 편향 (예측 − 기준치, 95% CI) | 절대 오차 평균 |", "|---|---|---|"]
  out += [f"| {n} | {e[n].mean():+.0f} ± {ci(e[n]):.0f} | {np.abs(e[n]).mean():.0f} |" for n in NAMES]
  dd = np.abs(e["선형"]) - np.abs(e["MLP"])
  out += [f"| 절대 오차 짝지은 차이 (선형 − MLP) | {dd.mean():+.0f} ± {ci(dd):.0f} | MLP가 작은 상태 {int((dd > 0).sum())}/{len(dd)} |", ""]
  out += [f"변환 오차 (상태 {len(C)}개, 가드 전 c, 기준치 rollout {d['cfg']['conv_R']}개)", "",
          "| 모델 | 종류 | 평균 절대 차이 | 모델 c 평균 | 기준치 c 평균 | 기준치 c 표준오차 평균 (잡음 바닥) |", "|---|---|---|---|---|---|"]
  for n in NAMES:
    for kind, lab in (("buy", "주문자"), ("sup", "공급자")):
      c, m, s = (np.concatenate([np.asarray(x, float) for x in xs]) for xs in
                 ([r[n][kind] for r in C], [r[f"mc_{kind}"] for r in C], [r[f"se_{kind}"] for r in C]))
      out.append(f"| {n} | {lab} | {conv_err(c, m):.0f} | {c.mean():.0f} | {m.mean():.0f} | {s.mean():.0f} |")
  return "\n".join(out)


if __name__ == "__main__":
  cfg, t0 = Cfg(), time.time()
  if "value" in sys.argv:
    print(value_table(value(cfg)), flush=True)
  if "eval" in sys.argv:
    path = evaluate(cfg)
    print(diag.table(path, "근시안", "MLP 유지 가치", "선형 유지 가치", "규칙 기반 (0.3, 0.3)"), "\n")
    print(diag.table(path, "선형 유지 가치", "MLP 유지 가치"))
  print(f"total {time.time() - t0:.0f}s")
