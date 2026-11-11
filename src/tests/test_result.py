"""실험 스크립트 보조 함수 테스트: 중간 저장과 이어 쓰기.
중간 저장 파일은 같은 설정일 때만 이어 쓰고, 설정이 다르면 멈춘다.
저장한 기록이 그대로 돌아오는지 본다.
"""
from dataclasses import replace
import pytest
from utils.params import Cfg
from result.exp_long_x2 import save, part


# 저장한 기록이 그대로 돌아오고, 파일이 없으면 빈 목록, 설정이 다르면 멈춘다
def test_part(tmp_path):
  p, c = str(tmp_path / "a.part.json"), Cfg()
  assert part(p, c) == []
  rows = [dict(name="fixed", rep=3, Pi=[0.1, 1 / 3])]
  save(p, dict(cfg=repr(c), rows=rows))
  assert part(p, c) == rows
  with pytest.raises(AssertionError):
    part(p, replace(c, T=10))
