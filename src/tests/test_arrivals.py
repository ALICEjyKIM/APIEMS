"""인스턴스 생성과 기간별 도착 테스트.
시장 조건(공급 편중도, 대체 공급자 수, 주문당 품목 수)이 구조에 그대로 반영되는지 확인한다.
공급이 평균 수요를 감당하는지, 재현성·공통 난수·재참여 변동 크기를 확인한다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from utils.instance import make, new_buy, qty_range
from utils.arrivals import new_buyers, new_sups, perturb
from utils.common import rng, ret_u


# 재참여율 ret_ss로 쌓인 안정 상태 활동 주문자의 수요를 기대 실효 공급(품목별 총 공급용량 × 공급자 자리 점유율 occ)이 약 cover배로 감당한다 (k와 무관)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_supply_covers_steady_demand(k):
  cfg = replace(Cfg(), items_per_order=k)
  inst = make(cfg, 0)
  act, dem, burn, n = [], np.zeros(cfg.n_items), 50, 400
  for t in range(burn + n):
    act = [(i, b) for i, b in act if ret_u(cfg, 0, "buy", i, t) < cfg.ret_ss]
    act += [(t * 100 + m, b) for m, b in enumerate(new_buyers(cfg, inst, 0, t))]
    if t >= burn:
      dem += sum(b.qty for _, b in act)
  ratio = inst.sup_cap.sum(0) * cfg.occ / (dem / n)
  assert (ratio > 1).all()
  assert abs(ratio.mean() - cfg.cover) < 0.2


# 주문 한 건의 품목당 평균 수량이 품목별 총 공급용량의 1/4 이하다 (품목당 수량이 가장 큰 k = 1 포함)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_order_small_vs_capacity(k):
  cfg = replace(Cfg(), items_per_order=k)
  lo, hi = qty_range(cfg)
  assert (lo + hi) / 2 <= make(cfg, 0).sup_cap.sum(0).min() / 4


# 공급 편중도를 바꿔도 공급자 수, 대체 공급자 수, 품목별 총 공급용량, 신규 주문(기대 수요)이 같다
def test_conc_keeps_market_totals():
  cfg = Cfg()
  insts = [make(replace(cfg, conc=c), 0) for c in (0.0, 0.5, 1.0)]
  for inst in insts:
    assert len(inst.sups) == cfg.n_sup
    assert (inst.sup_items.sum(0) == cfg.n_alt).all()
    assert np.array_equal(inst.sup_cap.sum(0), insts[0].sup_cap.sum(0))
    for t in range(10):
      a, b = new_buyers(cfg, inst, 0, t), new_buyers(cfg, insts[0], 0, t)
      assert [p.qty.tolist() for p in a] == [p.qty.tolist() for p in b]
      assert [p.price.tolist() for p in a] == [p.price.tolist() for p in b]


# 주문당 품목 수만 바꾸면 공급자 수, 공급자별 공급 품목, 품목별 총 공급용량, 기간당 신규 주문자 수가 같다
def test_items_per_order_keeps_supply():
  cfgs = [replace(Cfg(), items_per_order=k) for k in (1, 2, 3)]
  insts = [make(c, 0) for c in cfgs]
  for c, inst in zip(cfgs, insts):
    assert len(inst.sups) == len(insts[0].sups)
    assert np.array_equal(inst.sup_items, insts[0].sup_items)
    assert np.array_equal(inst.sup_cap.sum(0), insts[0].sup_cap.sum(0))
    assert [len(new_buyers(c, inst, 0, t)) for t in range(10)] == [len(new_buyers(cfgs[0], insts[0], 0, t)) for t in range(10)]


# 공급 편중도 0이면 모든 공급자의 품목 수가 같다
@pytest.mark.parametrize("n_alt", [1, 2, 3])
def test_conc_zero_equal_degrees(n_alt):
  d = make(replace(Cfg(), conc=0.0, n_alt=n_alt), 0).sup_items.sum(1)
  assert (d == d[0]).all()


# 공급 편중도를 올리면 공급자별 품목 수의 최댓값과 분산이 커진다
@pytest.mark.parametrize("n_alt", [2, 3])
def test_conc_increases_max_and_var(n_alt):
  ds = [make(replace(Cfg(), conc=c, n_alt=n_alt), 0).sup_items.sum(1) for c in (0.0, 0.5, 1.0)]
  assert ds[0].max() < ds[1].max() < ds[2].max()
  assert ds[0].var() < ds[1].var() < ds[2].var()


# 모든 품목의 대체 공급자 수가 n_alt, 모든 공급자가 1품목 이상, 연결에만 용량이 있다
@pytest.mark.parametrize("n_alt", [1, 2, 3])
@pytest.mark.parametrize("conc", [0.0, 0.5, 1.0])
def test_structure_valid(n_alt, conc):
  inst = make(replace(Cfg(), n_alt=n_alt, conc=conc), 0)
  assert (inst.sup_items.sum(0) == n_alt).all()
  assert (inst.sup_items.sum(1) >= 1).all()
  assert (inst.sup_cap[~inst.sup_items] == 0).all()
  assert (inst.sup_cap[inst.sup_items] >= 1).all()


# 떠난 공급자 자리는 같은 품목 조합과 공급가능량의 새 공급자로 채워진다
def test_vacancy_refill_same_items():
  cfg = replace(Cfg(), p_sup_new=1.0)
  inst = make(cfg, 0)
  new = new_sups(cfg, inst, 0, 3, range(cfg.n_sup))
  assert set(new) == set(range(cfg.n_sup))
  for j, p in new.items():
    assert np.array_equal(p.items, inst.sup_items[j])
    assert np.array_equal(p.qty, inst.sup_cap[j])


# 모든 주문의 품목 수가 주문당 품목 수와 같다
@pytest.mark.parametrize("k", [1, 2, 3])
def test_items_per_order(k):
  cfg = replace(Cfg(), items_per_order=k)
  inst = make(cfg, 0)
  bs = [b for t in range(30) for b in new_buyers(cfg, inst, 0, t)]
  assert len(bs) > 0
  for b in bs:
    assert b.items.sum() == k
    lo, hi = qty_range(cfg)
    assert ((b.qty[b.items] >= lo) & (b.qty[b.items] <= hi)).all() and (b.qty[~b.items] == 0).all()
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
