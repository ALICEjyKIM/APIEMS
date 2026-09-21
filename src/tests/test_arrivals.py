"""인스턴스 생성과 기간별 도착 테스트.
시장 조건(다품목 공급자 비율, 대체 공급자 수, 주문당 품목 수)이 구조에 그대로 반영되는지 확인한다.
공급이 평균 수요를 감당하는지, 재현성·공통 난수·재참여 변동 크기를 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from utils.instance import make, new_buy
from utils.arrivals import new_buyers, new_sups, perturb
from utils.common import rng


# 평균적으로 품목별 총 공급용량이 신규 주문 수요보다 크다
def test_supply_covers_demand():
  cfg = Cfg()
  inst = make(cfg, 0)
  dem = np.zeros(cfg.n_items)
  n = 400
  for t in range(n):
    for b in new_buyers(cfg, inst, 0, t):
      dem += b.qty
  assert (dem / n < inst.sup_cap.sum(0)).all()


# 다품목 공급자 비율 0이면 모든 공급자가 1품목만 공급한다
def test_multi_ratio_zero_single_item():
  inst = make(replace(Cfg(), multi_ratio=0.0), 0)
  assert (inst.sup_items.sum(1) == 1).all()


# 다품목 공급자 비율을 올리면 다품목 공급자 수가 늘어난다
def test_multi_ratio_increases_multi_suppliers():
  cnt = [(make(replace(Cfg(), multi_ratio=r), 0).sup_items.sum(1) > 1).sum() for r in (0.0, 0.3, 0.6)]
  assert cnt[0] < cnt[1] < cnt[2]


# 모든 품목의 대체 공급자 수가 n_alt와 같다
@pytest.mark.parametrize("n_alt", [1, 2, 3])
@pytest.mark.parametrize("r", [0.0, 0.3, 0.6])
def test_n_alt_per_item(n_alt, r):
  inst = make(replace(Cfg(), n_alt=n_alt, multi_ratio=r), 0)
  assert (inst.sup_items.sum(0) == n_alt).all()
  assert (inst.sup_cap[~inst.sup_items] == 0).all()
  assert (inst.sup_cap[inst.sup_items] >= 1).all()


# 모든 주문의 품목 수가 주문당 품목 수와 같다
@pytest.mark.parametrize("k", [1, 2, 3])
def test_items_per_order(k):
  cfg = replace(Cfg(), items_per_order=k)
  inst = make(cfg, 0)
  bs = [b for t in range(30) for b in new_buyers(cfg, inst, 0, t)]
  assert len(bs) > 0
  for b in bs:
    assert b.items.sum() == k
    assert (b.qty[b.items] >= cfg.qty_lo).all() and (b.qty[~b.items] == 0).all()
    assert (b.price[b.items] > 0).all()


# 같은 seed·rep면 같은 결과, 다른 rep면 다른 결과
def test_reproducible():
  cfg = Cfg()
  a, b, c = make(cfg, 0), make(cfg, 0), make(cfg, 1)
  assert np.array_equal(a.base, b.base) and np.array_equal(a.sup_cap, b.sup_cap)
  assert not np.array_equal(a.base, c.base)
  x, y = new_buyers(cfg, a, 0, 3), new_buyers(cfg, a, 0, 3)
  assert [p.qty.tolist() for p in x] == [p.qty.tolist() for p in y]
  z = new_buyers(cfg, a, 1, 3)
  assert [p.qty.tolist() for p in x] != [p.qty.tolist() for p in z] or len(x) != len(z)


# 재참여 주문자의 주문은 품목 구성이 같고 수량·가격이 프로필과 가깝다
def test_perturb_close_to_profile():
  cfg = Cfg()
  inst = make(cfg, 0)
  profs = new_buy(cfg, inst, rng(cfg, 0, "arr", 99), 200)
  dq, dp = [], []
  for i, p in enumerate(profs):
    q = perturb(cfg, 0, "buy", i, 1, p)
    assert np.array_equal(q.items, p.items)
    assert (q.qty[q.items] >= 1).all()
    dq.append(np.abs(q.qty - p.qty)[p.items].sum() / p.qty.sum())
    dp.append(np.abs(q.price[p.items] / p.price[p.items] - 1).mean())
  assert np.mean(dq) < 3 * cfg.noise_qty
  assert np.mean(dp) < 3 * cfg.noise_price
  assert np.mean(dp) > 0


# 공통 난수: 기간 t의 신규 도착은 이전 기간과 빈 자리 구성에 무관하다
def test_arrivals_common_random():
  cfg = Cfg()
  inst = make(cfg, 0)
  a = new_buyers(cfg, inst, 0, 5)
  for t in range(5):
    new_buyers(cfg, inst, 0, t)
  b = new_buyers(cfg, inst, 0, 5)
  assert [p.qty.tolist() for p in a] == [p.qty.tolist() for p in b]
  n = len(inst.sups)
  full = new_sups(cfg, inst, 0, 5, range(n))
  part = new_sups(cfg, inst, 0, 5, [0, n - 1])
  for j in part:
    assert np.array_equal(part[j].price, full[j].price)
  assert set(part) == set(full) & {0, n - 1}
