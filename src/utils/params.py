"""실험 설정을 한곳에 모은 frozen Cfg.
기준 시장(이전 PROC15H와 같은 시장), 재참여, 청산(Chu IPB), 후보, 솔버 설정을 담는다.
값의 근거와 상태(확정·잠정)는 ref/decisions.md에 있고, 바꿀 때는 dataclasses.replace로 새 Cfg를 만든다.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
  """모든 수치 설정.
  시장 생성 값은 이전 PROC15H와 같아 같은 seed에서 같은 시장이 나온다 (tests/test_market.py).
  잠정 값은 ref/decisions.md의 "대기" 항목이 정해지면 바꾼다.
  """
  # 규모
  seed: int = 0
  T: int = 30
  reps: int = 30
  n_items: int = 15
  n_sup: int = 60  # 공급자 자리 수 = n_items × n_alt: 자리마다 품목 하나 (Chu Theorem 5의 단일 품목 공급자, 기간별 진실성)
  n_sup_multi: int = 24  # 다품목 비교 조건의 자리 수 (이전 PROC15H 시장, 자리당 평균 2.5품목)
  # 시장 구조: 품목당 평균 대체 공급자 n_alt, 공급자 자리 품목 수의 편중도 conc(단일 품목이면 효과 없음), 품목별 대체 공급자 수 변동 het
  n_alt: int = 4
  conc: float = 0.5
  het: float = 0.5
  items_per_order: int = 2
  dir_alpha: float = 5.0  # 품목 용량을 공급자에게 나누는 Gamma 형태 (클수록 고르게)
  cover: float = 1.3  # 실효 공급 / 명목 수요 (기준 자리 점유율로 나눠 명목 용량을 정한다)
  onboard: int = 2  # 떠난 공급자 자리를 새 후보가 채우기까지의 기간
  # 주문자
  lam_buy: float = 10.0  # 기간당 신규 주문자 평균 (Poisson)
  ret_ss: float = 0.5  # 명목 수요 계산용 명목 재참여율 (명목 활동 주문자 = lam_buy / (1 − ret_ss))
  qty_lo: int = 2
  qty_hi: int = 5
  qty_unit: int = 3
  k_ref: int = 2  # 품목당 수량 범위 [qty_lo, qty_hi] × qty_unit이 적용되는 기준 주문당 품목 수
  markup_lo: float = 1.1  # 주문자 가치 = 품목 기준가 × U(markup_lo, markup_hi)
  markup_hi: float = 1.5
  # 공급자 원가 = 품목 기준가 × U(cost_lo, cost_hi), 품목 기준가 U(base_lo, base_hi)
  cost_lo: float = 0.8
  cost_hi: float = 1.1
  base_lo: float = 10.0
  base_hi: float = 20.0
  # 재참여자의 수량·가치·원가 상대 변동
  noise_qty: float = 0.1
  noise_price: float = 0.05
  f_ship: float = 25.0  # 성립 주문당 허브 배송 고정비
  # 재참여 기본항: 세 점 (경험 없음·잉여 0) → r2_p0, (경험·잉여 0) → r2_p1, (경험·잉여율 기준점) → r2_p2
  # 기준점 r2_rate_*: 고정 padding에서 거래한 참여자 잉여율의 중앙값, 주문자·공급자 따로 (출발 점검, check_start_anchor_*.json)
  r2_p0: float = 0.5
  r2_p1: float = 0.7
  r2_p2: float = 0.85
  r2_rate_buy: float = 0.174
  r2_rate_sup: float = 0.085
  # 거래 이력 h ← h_rho·h + (1 − h_rho)·e, logit에 h_w·β2·(h − h0)
  h_rho: float = 0.8
  h_w: float = 1.0
  h0: float = 0.5
  # 기준점 대비 손실: r ← r_rho·r + (1 − r_rho)·x (거래할 때, 첫 거래는 r = x), G(d) = η·β1·d (d ≥ 0), max(η·λ·β1·d, −β2) (d < 0)
  r_rho: float = 0.8
  eta: float = 0.5
  lam_loss: float = 2.0
  # padding 제외 페널티 δ: 재참여 확률 0.5를 ex_p로 낮추는 logit 크기
  ex_p: float = 0.4
  # 청산(Chu IPB): Q_c = ⌈m × max(품목 c 최대 공급자 물량, 품목 c 한 주문 최대 요구량)⌉, 후보 = m_grid × theta_grid
  m_grid: tuple = (1.0, 1.1, 1.25, 1.5)
  m_check: tuple = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0)  # 규모 점검에서 같은 입찰로 비교하는 m (큰 시장의 이윤 최대 m이 1.5를 넘을 수 있음)
  theta_grid: tuple = (0.0, 0.25, 0.5)
  # 시나리오 예상 청산 (ref/decisions.md "예측 점검"): 과거 입찰 경험분포에서 뽑는 시나리오 수, 품목별 원가 표본이 이보다 적으면 전체 비율을 씀
  saa_k: int = 20
  saa_min: int = 5
  cbar: float = 0.05  # 재참여 보상 1인 상한 (공개 가중치 ω_i 대비)
  cpn_w: float = 1.0  # 재참여 보상 κ/ω를 재참여의 x에 넣는 비율 (민감도 0.5)
  best_fixed: tuple = (1.0, 0.0)  # 최선 고정값 (m, θ): 튜닝 반복 누적 이윤 평균 최대 (tune_fixed.json)
  a_max: float = 1e4  # 가상 공급자 단가 (Chu 6.1절): 모든 묶음 가치보다 커서 실제 배정에는 쓰이지 않고 padding 문제가 항상 풀린다
  # Q̂ 학습 (ref/decisions.md "Q̂ 학습 자료", "Q̂ 탐색 예산"): 학습 반복 rep0_train부터 n_train개, 그중 마지막 n_val개는 검증
  eps: float = 0.1  # 행동 정책이 후보를 무작위로 고를 확률 (나머지는 고정 padding)
  rep0_train: int = 2000
  n_train: int = 100
  n_val: int = 20
  ridge_grid: tuple = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
  mlp_layers: tuple = (2, 3)
  mlp_hidden: tuple = (32, 64)
  mlp_lr: tuple = (1e-3, 3e-3)
  mlp_epochs: int = 200
  mlp_batch: int = 256
  # TD 학습 (2차, ref/decisions.md "TD 학습"): 학습 반복 rep0_td부터 n_td개, 그중 마지막 n_hold개 반복은 MLP 조기 종료용
  rep0_td: int = 3000
  n_td: int = 300
  n_hold: int = 60
  n_proc: int = 20  # 병렬 작업자 수
  # 솔버 (gurobipy): VCG 차이를 정확히 쓰므로 MIPGap 0
  grb_seed: int = 0
  grb_threads: int = 1
  mip_gap: float = 0.0
  grb_numfocus: int = 3  # 수치 문제로 SUBOPTIMAL이면 이 NumericFocus로 한 번 더 푼다 (추정 입찰가의 동점 많은 청산)
  tol: float = 1e-6


# 시장 조건 (ref/design.md 7장): 기준 ×1에서 바꾸는 값
SCALE = {"x1": dict(), "x2": dict(lam_buy=20.0, n_sup=120, n_alt=8), "x4": dict(lam_buy=40.0, n_sup=240, n_alt=16)}
