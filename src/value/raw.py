"""원재료(result.train.fit_q3 collect) → 근사 방법별 학습 입력. 표 입력(value.feat.feats), 표 + 구조 요약(value.graph.feats_s), 그래프(value.graph.graph)를 한 번에 만든다.
adv: X, Xs, G, Y(고정 padding 대비 남은 누적 이윤 차이), rep, t, R0. td: X, Xs, G, P, Xn, Xsn, Gn, t, rep (마지막 기간의 다음 상태는 0 입력·빈 그래프 대신 None 표시).
근사 방법은 같은 상태·같은 반복을 받고 입력 형태만 다르다 (ref/decisions.md "질문 3 비교 범위").
"""
from glob import glob
import gzip
import pickle
import numpy as np
from env.sim import cands
from value.feat import feats
from value.graph import graph, feats_s

MODES = ("tab", "struct", "graph")


# 반복 파일 목록 (반복 번호 순)
def files(mode):
  return sorted(glob(f"result/data/raw/{mode}/*.pkl.gz"), key=lambda f: int(f.split("\\")[-1].split("/")[-1].split(".")[0]))


# 상태 하나의 (표 입력, 표 + 구조 요약, 그래프 목록)
def inputs(cfg, ob):
  o, P = ob
  return feats(cfg, o, P), feats_s(cfg, o, P), graph(cfg, o, P)


# adv 원재료 → 학습 자료
def load_adv(cfg, fs=None):
  X, Xs, G, Y, R, T, R0 = [], [], [], [], [], [], []
  for f in fs or files("adv"):
    d = pickle.load(gzip.open(f, "rb"))
    for row in d["rows"]:
      x, xs, g = inputs(cfg, row["obs"])
      X.append(x)
      Xs.append(xs)
      G.append(g)
      Y.append(np.array(row["pi"]) - row["pi"][0])
      R.append(d["rep"])
      T.append(row["t"])
      R0.append(row["pi"][0])
  return dict(X=np.array(X), Xs=np.array(Xs), G=G, Y=np.array(Y), rep=np.array(R), t=np.array(T), R0=np.array(R0))


# td 원재료 → 학습 자료 (다음 상태 입력은 후보별, 마지막 기간은 None)
def load_td(cfg, fs=None):
  X, Xs, G, P, Xn, Xsn, Gn, T, R = [], [], [], [], [], [], [], [], []
  for f in fs or files("td"):
    d = pickle.load(gzip.open(f, "rb"))
    for row in d["rows"]:
      x, xs, g = inputs(cfg, row["obs"])
      nx = [inputs(cfg, ob) if ob is not None else None for ob in row["nxt"]]
      X.append(x)
      Xs.append(xs)
      G.append(g)
      P.append(row["pi"])
      Xn.append([n[0] if n else np.zeros_like(x) for n in nx])
      Xsn.append([n[1] if n else np.zeros_like(xs) for n in nx])
      Gn.append([n[2] if n else None for n in nx])
      T.append(row["t"])
      R.append(d["rep"])
  return dict(X=np.array(X), Xs=np.array(Xs), G=G, P=np.array(P), Xn=np.array(Xn), Xsn=np.array(Xsn), Gn=Gn, t=np.array(T), rep=np.array(R))
