"""가치 근사 테스트 (슬라이스 3).
시장 요약 지표의 손계산 값, 참여자 하나를 뺄 때의 변화, 공급 편중도에 대한 불변성을 확인한다.
시뮬레이션 기준치, 학습 데이터, 선형 근사, 유지 가치 변환은 뒤 단계에서 이 파일에 더한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Obs, Env
from match.milp_solve import Policy
from utils.features import phi, drop, phi_drops

CFG = Cfg()
I = CFG.n_items


# 주문자 2명(품목 0을 10·4개, 제안가격 20·30 / 둘째는 품목 1도 2개, 제안가격 25), 공급자 2명(품목 0을 6개씩, 가격 10·12)
def two_obs():
  q, p = np.array([[10, 0], [4, 2]]), np.array([[20.0, 0], [30.0, 25.0]])
  cap, c = np.array([[6, 0], [6, 0]]), np.array([[10.0, 0], [12.0, 0]])
  return Obs(0, q, p, cap, c, [0, 1], [0, 1], np.array([0.5, 0.7]), np.array([0.6, 0.9]))


# 손계산: 개수, 품목별 총 요구량·공급가능량, 그 품목을 가진 참여자만의 평균 가격(없으면 0), 대체 공급자 수, 평균 재참여확률
def test_phi_hand():
  f = phi(two_obs())
  assert len(f) == 2 + 5 * 2 + 2
  assert f.tolist() == pytest.approx([2, 2, 14, 2, 12, 0, 25, 25, 11, 0, 2, 0, 0.6, 0.75])


# 주문자를 빼면 주문자 수 −1, 그 주문자의 품목 요구량만큼 감소, 평균 가격·재참여확률은 남은 주문자로 다시 계산
def test_drop_buyer_hand():
  o = two_obs()
  assert phi(drop(o, "buy", 1)).tolist() == pytest.approx([1, 2, 10, 0, 12, 0, 20, 0, 11, 0, 2, 0, 0.5, 0.75])
  assert drop(o, "buy", 1).bid == [0] and len(o.bid) == 2


# 공급자를 빼면 공급자 수 −1, 그 공급자의 공급가능량만큼 감소, 그 품목의 대체 공급자 수 −1
def test_drop_supplier_hand():
  assert phi(drop(two_obs(), "sup", 0)).tolist() == pytest.approx([2, 1, 14, 2, 6, 0, 25, 25, 12, 0, 1, 0, 0.6, 0.9])


# 실제 시장에서도 참여자 하나를 빼면 개수·총량·대체 공급자 수가 그 참여자만큼 바뀐다 (phi_drops 행 순서 = 주문자, 공급자)
def test_drop_real_market():
  env, pol = Env(CFG, 0), Policy(CFG, split=CFG.sh_ref)
  env.reset()
  for _ in range(3):
    env.step(pol.act(env.o))
  o, f = env.o, phi(env.o)
  D = phi_drops(o)
  assert D.shape == (len(o.bid) + len(o.sid), len(f))
  for k in range(len(o.bid)):
    d = f - D[k]
    assert d[0] == 1 and d[1] == 0 and np.array_equal(d[2:2 + I], o.q[k]) and (d[2 + I:2 + 2 * I] == 0).all()
  for k in range(len(o.sid)):
    d = f - D[len(o.bid) + k]
    assert d[1] == 1 and d[0] == 0 and np.array_equal(d[2 + I:2 + 2 * I], o.cap[k])
    assert np.array_equal(d[2 + 4 * I:2 + 5 * I], (o.cap[k] > 0).astype(float))


# 공급 편중도만 다른 두 시장(같은 rep)의 기간 0 지표는 품목별 평균 공급가격을 뺀 모든 값이 같다
# 평균 공급가격은 가격 수준이 공급자 속성이라 어느 공급자가 어느 품목을 맡는지(편중도가 정함)에 따라 달라진다
@pytest.mark.parametrize("rep", [0, 1])
def test_phi_conc_invariant(rep):
  fs = [phi(Env(replace(CFG, conc=c), rep).reset()) for c in (0.0, 0.5, 1.0)]
  keep = np.r_[0:2 + 3 * I, 2 + 4 * I:len(fs[0])]
  for f in fs[1:]:
    assert np.array_equal(f[keep], fs[0][keep])
