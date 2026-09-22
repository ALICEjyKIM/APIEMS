"""실험 설정을 한곳에 모은 frozen Cfg.
소규모 예비 실험의 규모, 시장 조건, 주문자·공급자 생성 범위를 담는다.
시장 조건을 바꿀 때는 dataclasses.replace(cfg, conc=...)를 쓴다.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
  """모든 수치 설정.
  규모, 시장 조건, 주문자·공급자 생성 범위, 재참여 변동.
  값을 바꾸려면 dataclasses.replace로 새 Cfg를 만든다.
  """
  # 규모
  seed: int = 0
  n_items: int = 6
  n_sup: int = 6
  T: int = 30
  reps: int = 10
  # 시장 조건
  conc: float = 0.5
  n_alt: int = 2
  items_per_order: int = 2
  # 주문자
  lam_buy: float = 10.0  # 기간당 신규 주문자 수 평균. 품목별 총 공급용량도 같은 비율로 커져 k = 1 주문 한 건(평균 21)이 용량(151)의 1/4 이하
  qty_lo: int = 2
  qty_hi: int = 5
  qty_unit: int = 3  # 수량 단위 배수 (공급용량도 같은 비율로 커짐)
  k_ref: int = 2  # 품목당 수량 범위 [qty_lo, qty_hi] × qty_unit이 적용되는 기준 주문당 품목 수
  markup_lo: float = 1.1
  markup_hi: float = 1.5
  # 공급자
  cost_lo: float = 0.8
  cost_hi: float = 1.1
  cover: float = 1.3  # 기대 실효 공급(품목별 총 공급용량 × 공급자 자리 점유율) / 안정 상태 수요
  occ: float = 0.6022222222222221  # 공급자 자리 점유율: 기준 몫 sh_ref로 운영한 튜닝용 반복의 실측값 (bench/tune.measure로 한 번 재서 고정)
  ret_ss: float = 0.5
  p_sup_new: float = 0.25  # 빈 공급자 자리 충원 확률 (빈자리 평균 4기간). 새 공급자 검증·계약·등록에 시간이 걸리고, 충원이 빠르면 공급자를 잃는 손해가 없어 유지 가치가 의미를 잃는다
  # 품목
  base_lo: float = 10.0
  base_hi: float = 20.0
  # 재참여 변동
  noise_qty: float = 0.1
  noise_price: float = 0.05
  # 재참여 반응: 잉여율(배분 잉여 / 제안 금액)의 로지스틱을 두 점으로 정한다 (주문자·공급자 동일):
  # 잉여 0이면 ret_p0, 기준 잉여율 r_ref(기준 몫 sh_ref로 마진의 30%를 받을 때의 평균 잉여율, 튜닝용 반복 실측)이면 ret_ref
  # 기울기 b = (logit(ret_ref) − logit(ret_p0)) / r_ref (bench/tune.measure)
  ret_p0: float = 0.5
  ret_ref: float = 0.85
  r_ref_buy: float = 0.10478922733208504
  r_ref_sup: float = 0.10198023148034958
  b_buy: float = 16.55323833900429
  b_sup: float = 17.00918923411489
  # 재참여확률 구간선형 근사: 잉여율 0 ~ 확률 pwl_hi 지점 등간격 pwl_n점 + 잉여율 pwl_rmax 끝점
  pwl_n: int = 5
  pwl_hi: float = 0.97
  pwl_rmax: float = 1.0
  # 고정 잉여 비율 (거래 마진 기준, 나머지는 플랫폼 몫). sh_ref = 사전 기준 몫(주문자, 공급자): 반응 보정의 기준 잉여율과 규칙 기반 튜닝의 비교 기준.
  # sh_buy·sh_sup = 규칙 기반이 쓰는 몫 (기본 시장 튜닝 결과)
  sh_ref: tuple = (0.3, 0.3)
  sh_buy: float = 0.3
  sh_sup: float = 0.3
  sh_buy_grid: tuple = (0.0, 0.1, 0.2, 0.3)
  sh_sup_grid: tuple = (0.1, 0.2, 0.3, 0.4)
  # 튜닝·보정용 반복 (평가 seed와 분리)
  tune_seed: int = 1
  n_tune: int = 10
  # 허브 통합 배송: 성립 주문당 허브→주문자 고정비만 (개당 비용·분할 공급 추가 비용 없음)
  f_ship: float = 25.0
  # MILP (gurobipy)
  tol: float = 1e-4  # 결정 검사 허용 오차 (잉여 범위, 플랫폼 이윤 ≥ 0). 솔버 정수 허용오차 1e-5가 마진을 거쳐 전파되는 크기보다 크게
  grb_seed: int = 0
  grb_threads: int = 1
  mip_gap: float = 1e-6
