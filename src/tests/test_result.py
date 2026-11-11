"""실험 스크립트 보조 함수 테스트: Cfg 키=값 덮어쓰기, 이후 손실 표 합치기, 중간 저장 이어 쓰기.
이후 손실 표는 m = 1 파일과 다른 m만 잰 파일을 같은 상태 · 같은 미래끼리 짝지어 합친다.
가짜 점검 파일 두 개로 손으로 계산한 값과 비교한다.
"""
import json
import pytest
from dataclasses import replace
from utils.params import Cfg
from result.exp_long_x2 import ovr, lbar, save, part

TS = (5, 12, 17, 22, 27)


# 튜플은 쉼표로 나눈 실수, 나머지는 원래 형
def test_ovr():
  c = ovr(Cfg(), dict(m_grid="1,1.05", T="10", ex_p="0.01", best_fixed="1.05,0"))
  assert c.m_grid == (1.0, 1.05) and c.T == 10 and c.ex_p == 0.01 and c.best_fixed == (1.05, 0.0)


# 구간마다 상태 하나: m = 1.1은 미래 2개 평균 (s, 3s) − 이번 기간 1 = 2s − 1, m = 1.05는 m = 1의 첫 미래와 짝지은 5s − 2
def test_lbar_merge(tmp_path):
  A = [dict(rep=0, t=t, I=[1, 2], R=[[10, 10 + s], [20, 20 + 3 * s]]) for s, t in enumerate(TS)]
  B = [dict(rep=0, t=t, I=[3], R=[[10 + 5 * s]]) for s, t in enumerate(TS)]
  pa, pb = tmp_path / "a.json", tmp_path / "b.json"
  json.dump(dict(cfg=repr(replace(Cfg(), m_grid=(1.0, 1.1))), rows=A), open(pa, "w"))
  json.dump(dict(cfg=repr(replace(Cfg(), m_grid=(1.05,))), rows=B), open(pb, "w"))
  assert lbar((1.0, 1.1), (pa,)) == [[0, 2 * s - 1] for s in range(5)]
  assert lbar((1.0, 1.05, 1.1), (pa, pb)) == [[0, 5 * s - 2, 2 * s - 1] for s in range(5)]


# 중간 저장: 저장한 기록이 그대로 돌아오고, 파일이 없으면 빈 목록, 설정이 다르면 멈춘다
def test_part(tmp_path):
  p, c = str(tmp_path / "a.part.json"), Cfg()
  assert part(p, c) == []
  rows = [dict(name="fixed", rep=3, Pi=[0.1, 1 / 3])]
  save(p, dict(cfg=repr(c), rows=rows))
  assert part(p, c) == rows
  with pytest.raises(AssertionError):
    part(p, replace(c, T=10))
