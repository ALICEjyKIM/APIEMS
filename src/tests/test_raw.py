"""원재료 수집·변환 테스트: 저장한 원재료에서 만든 표 입력이 온라인 feats와 같고, td의 Π와 adv의 목표가 직접 돌린 값과 같은지.
기준 시장 반복 2000을 T = 4로 짧게 돌려 임시 폴더에 저장한다.
Pool은 쓰지 않고 one을 직접 부른다.
"""
import gzip
import os
import pickle
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.sim import Sim, cands
from value.feat import feats
from value.raw import load_adv, load_td
from result import collect_raw
from result.collect import behave

CFG = replace(Cfg(), T=4)


# 임시 폴더에 adv·td 원재료를 하나씩 만든다
@pytest.fixture
def raw(tmp_path, monkeypatch):
  monkeypatch.setattr(collect_raw, "DIR", str(tmp_path))
  for mode in ("adv", "td"):
    os.makedirs(f"{tmp_path}/{mode}")
    collect_raw.one((CFG, mode, 2000))
  return tmp_path


# 원재료의 표 입력 = 온라인 feats, td의 Π = 직접 한 기간 진행한 Π, adv 목표의 고정 padding 열은 0
def test_raw_matches_online(raw):
  A = load_adv(CFG, [f"{raw}/adv/2000.pkl.gz"])
  D = load_td(CFG, [f"{raw}/td/2000.pkl.gz"])
  assert A["X"].shape[0] == D["X"].shape[0] == CFG.T - 1 and np.array_equal(A["X"], D["X"])
  assert (A["Y"][:, 0] == 0).all() and A["Xs"].shape[2] == A["X"].shape[2] + 66 and len(A["G"][0]) == len(CFG.m_grid)
  s, pol = Sim(CFG, 2000), behave(CFG, 2000)
  s.step(*pol(s.pre()))
  o = s.pre()
  assert np.array_equal(feats(CFG, o), D["X"][0])
  for j, a in enumerate(cands(CFG)):
    b = deepcopy(s)
    assert b.step(*a)["Pi"] == pytest.approx(D["P"][0, j])
    assert np.array_equal(feats(CFG, b.pre()), D["Xn"][0, j])
  assert all(g is None for g in D["Gn"][-1]) and (D["Xn"][-1] == 0).all()
  d = pickle.load(gzip.open(f"{raw}/td/2000.pkl.gz", "rb"))
  assert d["rows"][0]["obs"][1][1.0].y.dtype == np.int16
