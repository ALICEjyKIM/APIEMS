"""진단 원자료 저장·표 생성 테스트.
작은 설정으로 돌린 원자료가 json에 저장되고, 표의 누적 이윤이 원자료 합과 같은지 확인한다.
같은 상태를 다시 돌리면 같은 원자료가 나온다 (공통 난수, 고정 솔버 seed).
"""
import json
from dataclasses import replace
import numpy as np
from utils.params import Cfg
from match.milp_solve import Policy
from match.interface import rollout
from result.diag import run, table, per_rep, RUNS


# 원자료 저장: 정책·반복·기간 크기, 누적 이윤 = rollout 누적 이윤, 재실행 시 동일
def test_run_and_table(tmp_path, monkeypatch):
  monkeypatch.setattr("result.diag.RUNS", tmp_path)
  cfg = replace(Cfg(), T=4, reps=2)
  pol = {"근시안": lambda c: Policy(c), "규칙": lambda c: Policy(c, split=(0.3, 0.3))}
  path = run(cfg, pol, "test", "테스트")
  d = json.loads(path.read_text(encoding="utf-8"))
  assert path.name == "diag_test.json" and d["cfg"]["T"] == 4 and set(d["policies"]) == set(pol)
  for n, eps in d["policies"].items():
    assert len(eps) == 2 and all(len(e["profit"]) == 4 for e in eps)
    for rep, e in enumerate(eps):
      assert np.isclose(sum(e["profit"]), rollout(cfg, pol[n](cfg), rep)["profit"])
      assert np.isclose(per_rep(e, d["cfg"])["fill"], rollout(cfg, pol[n](cfg), rep)["fill"])
  s = table(path, "근시안", "규칙")
  assert "diag_test.json" in s and "짝지은 차이" in s and "t0–3" in s
  assert json.loads(run(cfg, pol, "test", "테스트").read_text(encoding="utf-8")) == d
