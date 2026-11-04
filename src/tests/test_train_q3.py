"""질문 3 학습 테스트: 네 방법이 같은 원재료에서 기간별 Q̂를 학습하고 adv 상태를 예측하며, 지표와 pickle 왕복이 맞는지.
원재료는 기준 시장 반복 2000·2001(td)과 2000(adv)을 T = 4로 짧게 돌려 임시 폴더에 만든다.
에폭을 줄인 Cfg로 돌리고 Pool은 쓰지 않는다.
"""
import os
import pickle
from dataclasses import replace
import numpy as np
import pytest
import torch
from utils.params import Cfg
from value.raw import load_adv, load_td
from result import collect_raw
from result.train_q3 import METHODS, fqi, predict, metrics, grid

torch.set_num_threads(2)
CFG = replace(Cfg(), T=4, mlp_epochs=20)


# 임시 폴더의 원재료 → (td 자료, adv 자료)
@pytest.fixture(scope="module")
def data(tmp_path_factory):
  d = tmp_path_factory.mktemp("raw")
  collect_raw.DIR = str(d)
  for mode, reps in (("td", (2000, 2001)), ("adv", (2000,))):
    os.makedirs(f"{d}/{mode}")
    for r in reps:
      collect_raw.one((CFG, mode, r))
  return load_td(CFG, [f"{d}/td/2000.pkl.gz", f"{d}/td/2001.pkl.gz"]), load_adv(CFG, [f"{d}/adv/2000.pkl.gz"])


# 네 방법 모두: 기간 1~3의 Q̂, adv 예측이 유한하고 목표 범위 안, 지표 키, pickle 왕복 후 같은 예측
@pytest.mark.parametrize("name", METHODS)
def test_fit_predict(data, name):
  D, A = data
  hold = D["rep"] == 2001
  h = next(h for n, h in grid(CFG) if n == name)
  h = dict(h, H=8) if "H" in h else h
  q = fqi(CFG, D, hold, name, h)
  assert sorted(q.qs) == [1, 2, 3]
  v = predict(q, name, A)
  assert v.shape == A["Y"].shape and np.isfinite(v).all()
  assert all((v[A["t"] == s] >= q.lo[s] - 1e-9).all() and (v[A["t"] == s] <= q.hi[s] + 1e-9).all() for s in (1, 2, 3))
  m = metrics(v, A["Y"])
  assert set(m) == {"val", "regret", "hit", "rho", "rmse", "pick0"} and m["regret"] >= 0 and 0 <= m["hit"] <= 1
  assert np.allclose(predict(pickle.loads(pickle.dumps(q)), name, A), v)


# 지표 정의: 사후 최선을 고르면 regret 0·적중 1, 순위가 반대면 rho −1
def test_metrics():
  Y = np.array([[0.0, 3.0, 1.0], [0.0, -1.0, 2.0]])
  m = metrics(Y.copy(), Y)
  assert m["regret"] == 0 and m["hit"] == 1 and m["val"] == pytest.approx(2.5) and m["rho"] == pytest.approx(1.0)
  assert metrics(-Y, Y)["rho"] == pytest.approx(-1.0)
