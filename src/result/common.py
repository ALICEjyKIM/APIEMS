"""실험 스크립트(exp, train, check)가 같이 쓰는 함수와 상수.
Cfg 키=값 덮어쓰기, 중간 저장 · 이어 쓰기, 기간 구간과 이후 손실 표, 반복 단위 신뢰구간, 미래 난수 번호, 점검 반복, 행동 정책, 실행 기록 요약.
옮기기 전 위치는 exp_long_x2 · check_dev_ci · check_state · headroom · check_pred · collect · main1이다 (ref/decisions.md 2026-11-16 "result 폴더 정리").
"""
from dataclasses import replace
import json
import os
import re
import numpy as np
from scipy.stats import t as tdist
from utils.rng import rng
from env.sim import cands

BINS = (0, 10, 15, 20, 25)  # 기간 구간 경계 (구간의 왼쪽 끝)
DEV = "result/runs/check_dev_x2.json"  # 이후 손실 표의 기본 출처 (한 번 올린 대가 점검 ×2)
COM = ("nb", "ns", "ret_b", "ret_s", "tight", "thin")  # 상태별 이후 손실 점검의 입찰 전 피처
REPS, T0 = range(1000, 1010), 5  # 점검 반복과 첫 기간 (예측 점검, 한 번 올린 대가 점검과 같은 상태)
K = 8  # 미래 난수 수


# 키=값 문자열로 바꾼 Cfg (튜플은 쉼표로 나눈 실수)
def ovr(cfg, ov):
  f = lambda k, v: tuple(float(x) for x in v.split(",")) if isinstance(getattr(cfg, k), tuple) else type(getattr(cfg, k))(v)
  return replace(cfg, **{k: f(k, v) for k, v in ov.items()})


# 점검 결과 파일이 잰 m 목록 (저장된 cfg 기록에서)
def grid(j):
  return tuple(float(x) for x in re.search(r"m_grid=\(([^)]*)\)", j["cfg"]).group(1).split(",") if x.strip())


# 이후 손실 표: 기간 구간 [BINS[k], BINS[k+1])마다 m별 평균 (D − I), m = 1 대비 (기간 0~4는 첫 구간의 자료 t = 5~9로 정해진다).
# 첫 파일이 m = 1을 담는다. 다른 파일의 m은 같은 상태에서 m = 1의 같은 미래 앞 k개와 짝짓는다
def lbar(ms, paths=(DEV,)):
  J = [json.load(open(p)) for p in paths]
  one, v = {(r["rep"], r["t"]): r for r in J[0]["rows"]}, {}
  assert grid(J[0])[0] == 1.0
  for j in J:
    assert {(r["rep"], r["t"]) for r in j["rows"]} == set(one)
    for i, m in enumerate(grid(j)):
      assert m not in v
      v[m] = [(r["t"], float(np.mean(np.array(r["R"])[:, i] - np.array(one[(r["rep"], r["t"])]["R"])[:len(r["R"]), 0]) - (r["I"][i] - one[(r["rep"], r["t"])]["I"][0]))) for r in j["rows"]]
  hi = lambda k: BINS[k + 1] if k + 1 < len(BINS) else 10 ** 9
  return [[float(np.mean([x for t, x in v[m] if lo <= t < hi(k)])) for m in ms] for k, lo in enumerate(BINS)]


# 중간 저장: 임시 파일에 다 쓴 뒤 바꿔 끼워, 쓰는 도중 컴퓨터가 꺼져도 깨진 파일이 남지 않게 한다
def save(path, d):
  with open(path + ".tmp", "w") as f:
    json.dump(d, f)
    f.flush()
    os.fsync(f.fileno())
  os.replace(path + ".tmp", path)


# 중간 저장된 실행 기록 (같은 설정으로 다시 실행할 때만 이어 쓴다)
def part(path, cfg):
  if not os.path.exists(path):
    return []
  j = json.load(open(path))
  assert j["cfg"] == repr(cfg)
  return j["rows"]


# 평균과 반복 군집 95% 구간 반폭 (군집 수 G, 자유도 G − 1)
def cci(v, g):
  u, n = np.unique(g), len(v)
  e = np.array([(v[g == k] - v.mean()).sum() for k in u])
  se = np.sqrt(len(u) / (len(u) - 1) * (e ** 2).sum()) / n
  return dict(mean=float(v.mean()), ci=float(tdist.ppf(0.975, len(u) - 1) * se), ci_iid=float(1.96 * v.std(ddof=1) / np.sqrt(n)), n=int(n), G=int(len(u)))


# 반복 jackknife 95% 구간: 통계 f(포함할 상태 마스크)를 반복 하나씩 빼고 다시 계산
def jack(f, g):
  u = np.unique(g)
  v = np.array([f(g != k) for k in u])
  se = np.sqrt((len(u) - 1) / len(u) * ((v - v.mean()) ** 2).sum())
  return dict(mean=float(f(np.ones(len(g), bool))), ci=float(tdist.ppf(0.975, len(u) - 1) * se))


# 최소제곱 계수
def ls(X, y):
  return np.linalg.lstsq(X, y, rcond=None)[0]


# 미래 k의 난수 번호 (다른 반복과 겹치지 않게)
def fut(rep, k):
  return 10 ** 6 + rep * K + k


# 행동 정책: 확률 eps로 후보 균등 무작위, 나머지는 고정 padding (난수는 반복·기간으로 고정)
def behave(cfg, rep):
  C = cands(cfg)
  def pol(o):
    g = rng(cfg, rep, "explore", o.t)
    return C[g.integers(len(C))] if g.random() < cfg.eps else C[0]
  return pol


# 기간별 기록 R의 요약
def rec(name, rep, R):
  a = lambda k, f=float: [f(r[k]) for r in R]
  return dict(name=name, rep=rep, Pi=a("Pi"), G=a("G"), cpn=a("cpn"), paid=a("paid"), sw=a("sw"), m=a("m"), th=a("th"), win=a("win", int),
              nb=a("nb", int), ns=a("ns", int), Ub=[int(r["Ub"].sum()) for r in R], Uj=[int(r["Uj"].sum()) for r in R])
