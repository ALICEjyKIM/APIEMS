"""Cfg와 공통 난수 스트림 테스트.
설정이 고정(frozen)되어 있고 소규모 예비 실험 범위 안에 있는지 확인한다.
같은 키면 같은 난수, 반복·스트림이 다르면 다른 난수인지 확인한다.
"""
import dataclasses
import pytest
from utils.params import Cfg
from utils.common import rng, ret_u


# Cfg는 값을 바꿀 수 없다
def test_cfg_frozen():
  cfg = Cfg()
  with pytest.raises(dataclasses.FrozenInstanceError):
    cfg.T = 10


# 규모와 시장 조건 기본값이 예비 실험 범위 안에 있다
def test_cfg_scale():
  cfg = Cfg()
  assert (cfg.n_items, cfg.n_sup, cfg.n_alt) == (6, 6, 2)
  assert 2 <= cfg.items_per_order <= 3
  assert 20 <= cfg.T <= 30
  assert cfg.reps == 30
  assert 0 <= cfg.conc <= 1
  assert cfg.n_alt <= cfg.n_sup <= cfg.n_items * cfg.n_alt


# 같은 키는 같은 난수, 반복·스트림·키가 다르면 다른 난수
def test_rng_keys():
  cfg = Cfg()
  a = rng(cfg, 0, "arr", 3).random(5)
  assert (a == rng(cfg, 0, "arr", 3).random(5)).all()
  assert (a != rng(cfg, 1, "arr", 3).random(5)).any()
  assert (a != rng(cfg, 0, "noise", 3).random(5)).any()
  assert (a != rng(cfg, 0, "arr", 4).random(5)).any()


# 재참여 난수는 참여자·기간별로 고정된 균등난수
def test_ret_u():
  cfg = Cfg()
  u = ret_u(cfg, 0, "buy", 7, 2)
  assert 0 <= u < 1
  assert u == ret_u(cfg, 0, "buy", 7, 2)
  assert u != ret_u(cfg, 0, "sup", 7, 2)
  assert u != ret_u(cfg, 0, "buy", 7, 3)
