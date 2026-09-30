"""슬라이스 4 진단: 학습 타깃의 잡음과 미래 rollout 평균 수 M의 효과.
같은 상태에서 미래 키만 바꾼 rollout K개를 돌려 상태 내 분산(잡음)과 상태 간 분산(신호)을 나누고,
M = 1, 2, 5, 10의 표준오차·달성 가능 R² 상한·반쪽 신뢰도와 계산비용을 함께 재서 본 실험의 M을 제안한다.
"""
import copy
import json
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg, SCALE15
from env.platform import Env
from bench.rule_split import rule
from match.mc_value import fkey
from vfa.train import collect_obs
from result import diag

OUT = diag.RUNS / "diag_s4_target.json"
STATE_REPS = 10
STATE_T = (5, 10, 15, 20)  # 5기간 간격이라 미래 키 K개(K ≤ 5 × mc_R)가 상태끼리 겹치지 않는다
K = 20
MS = (1, 2, 5, 10)


# 확정 환경 (슬라이스 3·3b에서 고정)
def env_cfg(**kw):
  return replace(Cfg(), **{**SCALE15, **kw})


# 상태 하나에서 미래 키 k의 후속 정책 vf_H기간 이윤 합 (mc_value와 같은 방식, R을 K까지 늘린다)
def futures(env, k_max):
  cfg, out = env.cfg, []
  for k in range(k_max):
    e = copy.deepcopy(env)
    e.rep = fkey(cfg, env.rep, env.t, k)
    pol = rule(cfg)
    out.append(sum(e.step(pol.act(e.o))["profit"] for _ in range(cfg.vf_H)))
  return np.array(out)


# 상태 × 미래 행렬: 튜닝 seed 반복을 규칙 기반으로 돌리며 STATE_T 기간에서 미래 K개
def matrix(cfg, reps=STATE_REPS, ts=STATE_T, k_max=K):
  tc = replace(cfg, seed=cfg.tune_seed)
  V, meta, t0 = [], [], time.time()
  for rep in range(reps):
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    for t in range(max(ts) + 1):
      if t in ts:
        V.append(futures(env, k_max))
        meta.append(dict(rep=rep, t=t))
      env.step(pol.act(env.o))
  return np.array(V), meta, time.time() - t0


# 분산 분해: 상태 내 평균 분산 = 잡음, 상태 평균의 분산에서 잡음/K를 빼면 신호
def decompose(V):
  within = float(V.var(1, ddof=1).mean())
  between = float(V.mean(1).var(ddof=1))
  return within, between, max(between - within / V.shape[1], 0.0)


# M개 평균의 달성 가능 R² 상한 (exp3_diag와 같은 정의): 1 − (잡음/M) / (신호 + 잡음/M)
def ceiling(sig, noise, m, y_var=None):
  v = (y_var - noise if y_var is not None else sig) + noise / m
  return float(1 - (noise / m) / v)


# 반쪽 신뢰도: 미래를 두 묶음으로 나눠 각 M개씩 평균한 두 값의 상관 (독립 표본이므로 재현성 측정)
def split_half(V, m, seed=0):
  g = np.random.default_rng(seed)
  a, b = [], []
  for row in V:
    idx = g.permutation(len(row))
    a.append(row[idx[:m]].mean())
    b.append(row[idx[m:2 * m]].mean())
  a, b = np.array(a), np.array(b)
  return float(np.corrcoef(a, b)[0, 1]), float(np.mean(np.abs(a - b))), float(a.std(ddof=1))


NOTE = ("슬라이스 4 진단. 확정 환경(SCALE15, dir_alpha 5, cover 1.3, 재보정 occ·r_ref)에서 "
        f"튜닝 seed {STATE_REPS}반복 × 기간 {STATE_T} = 상태 {STATE_REPS * len(STATE_T)}개, 상태마다 미래 키 {K}개를 돌렸다. "
        "잡음 = 상태 내 평균 분산, 신호 = 상태 평균 분산 − 잡음/K. R² 상한 정의는 exp3_diag와 같다: "
        "1 − (잡음/M) / (y_var − 잡음 + 잡음/M). y_var는 실제 학습 데이터(vf_reps3반복)의 타깃 분산이다. "
        "반쪽 신뢰도는 미래를 두 묶음으로 나눠 M개씩 평균한 두 값의 상관이다.")

if __name__ == "__main__":
  t0 = time.time()
  cfg = env_cfg(vf_reps=Cfg().vf_reps3)
  print("A 상태 × 미래 행렬", flush=True)
  V, meta, sec_mat = matrix(cfg)
  within, between, sig = decompose(V)
  print(f"   상태 {V.shape[0]}개 × 미래 {V.shape[1]}개, {sec_mat:.0f}s "
        f"| 잡음 sd {np.sqrt(within):.0f} 신호 sd {np.sqrt(sig):.0f}", flush=True)
  print("B 실제 학습 데이터의 타깃 분산", flush=True)
  tcol = time.time()
  O, y, g = collect_obs(cfg)
  tcol = time.time() - tcol
  y_var = float(y.var())
  print(f"   학습 표본 {len(y)}개, 목표 sd {np.sqrt(y_var):.0f}, 수집 {tcol:.0f}s", flush=True)
  sec_roll = sec_mat / (V.shape[0] * V.shape[1])
  n_states = cfg.vf_reps * (cfg.T - cfg.vf_H + 1)
  rows = []
  for m in MS:
    r, mad, sd = split_half(V, m)
    rows.append(dict(M=m, se=float(np.sqrt(within / m)), se_over_signal_sd=float(np.sqrt(within / m) / np.sqrt(sig)),
                     ceil_local=ceiling(sig, within, m), ceil=ceiling(sig, within, m, y_var),
                     split_half_r=r, split_half_mad=mad, split_half_sd=sd,
                     solves=int(n_states * m * cfg.vf_H), sec=float(n_states * m * sec_roll),
                     sec_per_cell_hours=float(n_states * m * sec_roll / 3600)))
  res = dict(note=NOTE, cfg=asdict(cfg), scale15=SCALE15, states=V.shape[0], K=K, meta=meta,
             within_var=within, between_var=between, signal_var=sig, y_var=y_var, n_train=len(y),
             sec_per_rollout=sec_roll, collect_sec=tcol, n_states_per_cell=n_states,
             within_by_t={str(t): float(np.mean([V[i].var(ddof=1) for i, mm in enumerate(meta) if mm["t"] == t])) for t in STATE_T},
             rows=rows, V=V.tolist(), sec=time.time() - t0)
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
