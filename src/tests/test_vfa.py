"""가치 근사 테스트 (슬라이스 3).
시장 요약 지표의 손계산 값, 참여자 하나를 뺄 때의 변화, 공급 편중도에 대한 불변성을 확인한다.
시뮬레이션 기준치(결정성, 공통 난수), 학습 데이터, 선형 근사, 유지 가치 변환을 확인한다.
"""
import copy
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Obs, Env
from match.milp_solve import Policy
from utils.features import phi, drop, phi_drops
from match.mc_value import mc_value, fkey
from match.interface import rollout
from vfa.train import collect, val_mask
from vfa.linear import Linear, select

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


# 규칙 기반으로 몇 기간 돌린 환경 (시뮬레이션 기준치 테스트용, 짧은 호라이즌·적은 rollout)
def mid_env(t=3):
  cfg = replace(CFG, vf_H=3, mc_R=3)
  env, pol = Env(cfg, 0), Policy(cfg, split=(cfg.sh_buy, cfg.sh_sup))
  env.reset()
  for _ in range(t):
    env.step(pol.act(env.o))
  return env


# 환경에서 참여자를 빼면 관측이 지표의 drop과 같다 (주문자는 주문 삭제, 공급자는 자리를 비움)
@pytest.mark.parametrize("kind,k", [("buy", 0), ("buy", 2), ("sup", 1)])
def test_env_drop_matches_feature_drop(kind, k):
  env = mid_env()
  o = env.o
  env.drop(kind, k)
  assert np.array_equal(phi(env.o), phi(drop(o, kind, k)))
  assert env.o.bid == drop(o, kind, k).bid and env.o.sid == drop(o, kind, k).sid


# 시뮬레이션 기준치: 결정적이고 원래 환경을 바꾸지 않으며, k번째 값은 미래 키 k로 직접 돌린 vf_H기간 이윤 합이고 미래마다 다르다
def test_mc_value_manual():
  env = mid_env()
  t, bid = env.t, list(env.o.bid)
  v = mc_value(env)
  assert len(v) == env.cfg.mc_R and np.array_equal(v, mc_value(env))
  assert env.t == t and env.o.bid == bid
  e = copy.deepcopy(env)
  e.rep = fkey(env.cfg, env.rep, env.t, 1)
  pol = Policy(env.cfg, split=(env.cfg.sh_buy, env.cfg.sh_sup))
  assert v[1] == pytest.approx(sum(e.step(pol.act(e.o))["profit"] for _ in range(env.cfg.vf_H)))
  assert len(set(v.round(6))) > 1


# 공통 난수: 참여자를 뺀 상태의 기준치는 미리 뺀 환경의 기준치와 같다 (같은 미래 키)
def test_mc_value_drop_common_random():
  env = mid_env()
  e = copy.deepcopy(env)
  e.drop("sup", 0)
  assert np.array_equal(mc_value(env, ("sup", 0)), mc_value(e))


# 선형 근사: 잡음 없는 선형 목표를 λ = 0으로 정확히 복원하고, 표준편차 0인 지표가 있어도 맞춘다. λ가 크면 가중치가 줄어든다
def test_linear_recovers():
  g = np.random.default_rng(0)
  X = np.c_[g.normal(size=(50, 4)) * [1, 10, 100, 0.1], np.full(50, 7.0)]
  y = X[:, :4] @ [2.0, -1.0, 0.5, 30.0] + 3.0
  m = Linear(0.0).fit(X, y)
  assert m.predict(X) == pytest.approx(y) and m.predict(X[:3] * 2) == pytest.approx((X[:3] * 2)[:, :4] @ [2.0, -1.0, 0.5, 30.0] + 3.0)
  assert np.linalg.norm(Linear(100.0).fit(X, y).w) < np.linalg.norm(m.w)


# λ 선택: 후보마다 공통 검증 분할(뒤 vf_val 비율 반복)의 평균제곱오차를 재고, 가장 작은 λ로 전체 데이터를 다시 맞춘다
def test_select_lambda():
  g = np.random.default_rng(1)
  X = g.normal(size=(300, 5))
  y = X @ g.normal(size=5) + g.normal(size=300) * 3
  reps = np.repeat(np.arange(30), 10)
  m, mse = select(CFG, X, y, reps)
  va = val_mask(CFG, reps)
  assert set(mse) == set(CFG.lin_lams) and va.sum() == 60 and (reps[va] >= 24).all()
  assert m.lam == min(mse, key=mse.get)
  assert mse[m.lam] == pytest.approx(np.mean((Linear(m.lam).fit(X[~va], y[~va]).predict(X[va]) - y[va]) ** 2))
  assert m.predict(X) == pytest.approx(Linear(m.lam).fit(X, y).predict(X))


# 학습 데이터: 튜닝 seed 규칙 기반 rollout의 t ≤ T − vf_H 상태 지표와 이후 vf_H기간 이윤 합 (rollout 누적 이윤과 맞물림)
def test_collect():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=2)
  X, y, g = collect(cfg)
  assert X.shape == (2 * 5, len(phi(Env(cfg, 0).reset()))) and g.tolist() == [0] * 5 + [1] * 5
  tc = replace(cfg, seed=cfg.tune_seed)
  env = Env(tc, 0)
  assert np.array_equal(X[0], phi(env.reset()))
  pol, pr = Policy(tc, split=(tc.sh_buy, tc.sh_sup)), []
  for _ in range(tc.T):
    pr.append(env.step(pol.act(env.o))["profit"])
  assert y[:5] == pytest.approx([pr[t] + pr[t + 1] for t in range(5)])
  assert sum(pr) == pytest.approx(rollout(tc, Policy(tc, split=(tc.sh_buy, tc.sh_sup)), 0)["profit"])
