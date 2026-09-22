"""실험 1 파이프라인 테스트.
작은 설정으로 value·eval 단계를 돌려 원자료 json의 구조, 요약, 표 생성을 확인한다.
eval 단계는 다시 맞춘 모델이 value 단계 적합과 같을 때만 진행한다.
"""
import json
from dataclasses import replace
import numpy as np
from utils.params import Cfg
from result import exp1_linear_vs_nn as exp1
from result import diag

SMALL = replace(Cfg(), T=6, reps=2, vf_H=2, vf_reps=5, mc_R=2, conv_R=3, nn_epochs=20)


# value → eval: 적합 기록(격자·검증 MSE·격자 끝), 예측·변환 원자료, 정책 비교 원자료와 요약(짝지은 차이 4종), 출처가 적힌 표
def test_exp1_pipeline(tmp_path, monkeypatch):
  monkeypatch.setattr(diag, "RUNS", tmp_path)
  vp = exp1.value(SMALL, pred_t=(1, 2), conv_t=(2,))
  d = json.loads(vp.read_text(encoding="utf-8"))
  for n in exp1.NAMES:
    assert len(d["fit"][n]["val_mse"]) == 5 and d["fit"][n]["edge"] == (d["fit"][n]["pick"] in (d["fit"][n]["grid"][0], d["fit"][n]["grid"][-1]))
  assert len(d["pred"]) == 2 and len(d["pred"][0]["mc"]) == SMALL.mc_R
  row = d["conv"][0]
  assert len(row["se_buy"]) == len(row["mc_buy"]) == len(row["MLP"]["buy"]) and len(row["mc_sup"]) == len(row["선형"]["sup"])
  s = exp1.value_table(vp)
  assert "exp1_value.json" in s and "격자 끝" in s and "선형 − MLP" in s
  ep = exp1.evaluate(SMALL)
  e = json.loads(ep.read_text(encoding="utf-8"))
  assert ep.name == "exp1_eval.json" and set(e["policies"]) == {"근시안", "규칙 기반 (0.3, 0.3)", "선형 유지 가치", "MLP 유지 가치"}
  assert set(e["summary"]["paired"]) == {f"{a} − {b}" for a, b in exp1.PAIRS}
  p = e["summary"]["paired"]["MLP 유지 가치 − 근시안"]
  mlp, my = (np.array([sum(x["profit"]) for x in e["policies"][n]]) for n in ("MLP 유지 가치", "근시안"))
  assert np.isclose(p["mean"], (mlp - my).mean()) and p["n"] == SMALL.reps
