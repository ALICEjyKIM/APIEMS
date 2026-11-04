"""시장 생성 테스트.
다품목 조건(n_sup_multi)에서 이전 구조(PROC15H)의 기준 자료(tests/data/legacy_market.npz)와 같은 값이 나오는지 확인한다.
기준 시장 통계(명목 수요·용량)와 공통 난수(같은 키 → 같은 값)도 확인한다.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from utils.params import Cfg
from utils.market import make, candidate, new_buyers, perturb, demand, capacity
from utils.rng import ret_u

REF = np.load(Path(__file__).parent / "data" / "legacy_market.npz")


# 기준 자료와 같은 시장·도착·후보·변동·재참여 난수
@pytest.mark.parametrize("rep", range(3))
def test_same_as_legacy(rep):
  cfg = replace(Cfg(), n_sup=Cfg().n_sup_multi)
  mk = make(cfg, rep)
  for k, v in dict(base=mk.base, w=mk.w, ntgt=mk.n_tgt, K=mk.K, deg=mk.deg,
                   sqty=np.array([s.qty for s in mk.sups]), sprice=np.array([s.price for s in mk.sups])).items():
    assert np.array_equal(v, REF[f"r{rep}_{k}"]), k
  for t in range(4):
    bs = new_buyers(cfg, mk, rep, t)
    assert np.array_equal(np.array([b.qty for b in bs]).reshape(-1, cfg.n_items), REF[f"r{rep}_t{t}_bqty"])
    assert np.array_equal(np.array([b.price for b in bs]).reshape(-1, cfg.n_items), REF[f"r{rep}_t{t}_bprice"])
    c = candidate(cfg, mk, rep, t, 3)
    assert np.array_equal(c.qty, REF[f"r{rep}_t{t}_cqty"]) and np.array_equal(c.price, REF[f"r{rep}_t{t}_cprice"])
    p = perturb(cfg, rep, "buy", 7, t, mk.sups[0])
    assert np.array_equal(p.qty, REF[f"r{rep}_t{t}_pqty"]) and np.array_equal(p.price, REF[f"r{rep}_t{t}_pprice"])
    u = [ret_u(cfg, rep, k, i, t) for k in ("buy", "sup") for i in range(5)]
    assert np.array_equal(u, REF[f"r{rep}_t{t}_ret"])


# 기준 시장 통계: 품목 명목 수요 28, 명목 용량 47.32, 연결 수 = 품목 수 × n_alt
def test_market_stats():
  cfg = Cfg()
  assert demand(cfg) == pytest.approx(28.0)
  assert capacity(cfg) == pytest.approx(47.32)
  mk = make(cfg, 0)
  assert mk.n_tgt.sum() == cfg.n_items * cfg.n_alt == mk.deg.sum()
  assert (np.array([s.items for s in mk.sups]).sum(0) == mk.n_tgt).all()


# 같은 키는 같은 난수, 다른 키는 다른 난수 (공통 난수)
def test_common_random_numbers():
  cfg = Cfg()
  mk = make(cfg, 0)
  a, b = new_buyers(cfg, mk, 0, 5), new_buyers(cfg, mk, 0, 5)
  assert len(a) == len(b) and all(np.array_equal(x.qty, y.qty) for x, y in zip(a, b))
  assert ret_u(cfg, 0, "buy", 3, 2) == ret_u(cfg, 0, "buy", 3, 2) != ret_u(cfg, 0, "buy", 3, 3)


# 기본 시장: 공급자 자리와 빈자리 후보가 모두 품목 하나만 판다 (Chu Theorem 5 조건)
def test_single_item_suppliers():
  cfg = Cfg()
  for rep in range(3):
    mk = make(cfg, rep)
    assert (mk.deg == 1).all() and all(s.items.sum() == 1 for s in mk.sups)
    assert all(candidate(cfg, mk, rep, t, j).items.sum() == 1 for t in range(3) for j in range(cfg.n_sup))
