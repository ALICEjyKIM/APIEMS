"""legacy 환경 재현 테스트 (조달 시장 v2 추가 후).
저장된 원자료(result/runs/*.json)와 같은 설정으로 다시 돌려 기간별 값이 그대로인지 확인한다.
Cfg 기본값과 SCALE15 칸은 market = "legacy"이며 v2 코드 경로를 타지 않는다.
"""
import json
import numpy as np
import pytest
from env.platform import Env, Env2
from env.response import Logistic, response
from match.mc_value import mc_value
from vfa.train import tkey
from utils.params import Cfg
from result import diag
from result.exp5_fair import cell_cfg, tag
from result.diag_s7_myopic import POLS, episode


# 기본 Cfg와 SCALE15 칸은 legacy이고 Env·반응 함수도 legacy 것을 쓴다
def test_legacy_defaults():
  for cfg in (Cfg(), cell_cfg()):
    assert cfg.market == "legacy"
    assert type(Env(cfg, 0)) is Env and isinstance(response(cfg), Logistic)


# 진단 s7 (SCALE15 기준 칸, Rule·Myopic·Myopic-sup, 평가 seed 반복 0)의 기간별 원자료가 저장값과 같다
@pytest.mark.parametrize("pol", list(POLS))
def test_diag_s7_reproduced(pol):
  src = json.loads((diag.RUNS / "diag_s7_myopic.json").read_text(encoding="utf-8"))["runs"][pol][0]
  out = episode(cell_cfg(), POLS[pol], 0)
  for k, v in src.items():
    assert np.allclose(out[k], v, equal_nan=True), k


# 실험 7 Myopic-sup 단위(반복 0)의 기간별 이윤·공급자 잉여가 저장값과 같다
def test_exp7_unit_reproduced():
  rows = json.loads((diag.RUNS / "exp7_units" / "alt4_conc0.5_k2_Myopic-sup_r0.json").read_text(encoding="utf-8"))["rows"]
  out = episode(cell_cfg(), POLS["Myopic-sup"], 0)
  assert np.allclose(out["profit"], [r["profit"] for r in rows])
  assert np.allclose(out["sb"], [r["sb"] for r in rows])


# 실험 5 시험 상태(반복 108, 기간 0)의 MC 미래가치 앞 2개가 저장값과 같다 (tune seed, 키 tkey)
def test_exp5_future_reproduced():
  from dataclasses import replace
  cfg = cell_cfg()
  src = json.loads((diag.RUNS / f"exp5_{tag(cfg)}.json").read_text(encoding="utf-8"))
  idx = src["test_t"].index(0)
  assert src["test_g"][idx] == cfg.test_rep0
  env = Env(replace(cfg, seed=cfg.tune_seed), cfg.test_rep0)
  env.reset()
  assert np.allclose(mc_value(env, R=2, key=tkey), src["futures"]["test"][idx][:2])


# v2 preset은 Env2를 만든다 (dispatch 확인)
def test_v2_dispatch():
  from dataclasses import replace
  from utils.params import PROC15
  cfg = replace(Cfg(), **PROC15)
  assert type(Env(cfg, 0)) is Env2


# PROC15B 재현 (이력 환경 추가 후): 진단 s9a 기준 칸 Myopic·Rule 5반복 평균 지표와 s9b 상태(base 반복 0 기간 8)의 미래 2개가 저장값과 같다
def test_proc15b_reproduced():
  from result.diag_s9_market import episode as ep9, BASE, POLS as P9, REPS
  from result.diag_s9_interaction import state, value
  src = json.loads((diag.RUNS / "diag_s9_market.json").read_text(encoding="utf-8"))["conds"]["base"]["metrics"]
  assert type(Env(BASE, 0)) is Env2
  for pn, mk in P9.items():
    eps = [ep9(BASE, mk, rep) for rep in REPS]
    for k in ("act_sup", "act_buy", "fill", "zero_all", "cap_ratio", "dep_sup"):
      assert np.mean([e[k] for e in eps]) == pytest.approx(src[pn][k], abs=1e-12), (pn, k)
  u = json.loads((diag.RUNS / "diag_s9_units" / "base_r0_t8_c0.json").read_text(encoding="utf-8"))
  assert np.allclose(value(state(BASE, 0, 8), [], 2), u["values"][""][:2])
