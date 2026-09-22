"""슬라이스 2 진단 상태의 원자료 저장.
환경 변경 전후 상태(용량 91·T 25, 용량 182·T 30, 충원 확률 0.25·몫 격자 튜닝, 두 점 반응·실측 점유율)를 같은 코드로 다시 돌려 반복별 원자료를 남긴다.
공통 난수와 고정 솔버 seed 덕에 당시 표가 그대로 복원된다. 실행: src/에서 python -m result.diag_slice2 [상태 태그...].
"""
import sys
from dataclasses import replace
from utils.params import Cfg
from match.milp_solve import Policy
from result.diag import run, table


# 근시안과 규칙 기반 몫 후보들 (이름 → cfg를 받아 정책을 만드는 함수)
def policies(splits):
  return {"근시안": lambda c: Policy(c), **{f"규칙 기반 {sp}": (lambda c, sp=sp: Policy(c, split=sp)) for sp in splits}}


OLD = policies([(0.3, s) for s in (0.1, 0.2, 0.3, 0.4)])
GRID = policies([(b, s) for b in Cfg().sh_buy_grid for s in Cfg().sh_sup_grid])
BEST = f"규칙 기반 {(Cfg().sh_buy, Cfg().sh_sup)}"
# 옛 상태는 당시 값으로 고정한다 (반응: 잉여 0이면 0.1, 기울기 보정값 / 용량: occ로 당시 용량 재현)
OLD_RESP = dict(ret_p0=0.1)
# 상태 태그 → (cfg, 설명, 정책, 짝지은 차이의 비교 정책)
STATES = {
  "s2a_cap91_T25": (replace(Cfg(), T=25, p_sup_new=0.5, occ=1.0, b_buy=39.55078125, b_sup=24.70703125, sh_buy=0.3, sh_sup=0.3, **OLD_RESP),
                    "슬라이스 2 종료 시점: 용량 91, T 25, p_sup_new 0.5, 옛 기준 보정 (39.55, 24.71)", OLD, "규칙 기반 (0.3, 0.3)"),
  "s2b_cap182_T30": (replace(Cfg(), T=30, p_sup_new=0.5, occ=0.5, b_buy=31.73828125, b_sup=35.25390625, sh_buy=0.3, sh_sup=0.3, **OLD_RESP),
                     "환경 변경 (B)(D) 후: 용량 182, T 30, p_sup_new 0.5, 옛 기준 보정 (31.74, 35.25)", OLD, "규칙 기반 (0.3, 0.3)"),
  "s2c_psup025_grid": (replace(Cfg(), occ=1 / 3, b_buy=22.55859375, b_sup=53.41796875, sh_buy=0.0, sh_sup=0.1, **OLD_RESP),
                       "p_sup_new 0.25, 용량 273, T 30, 기준 잉여율 보정, 몫 격자 튜닝 결과 (0.0, 0.1)", GRID, "규칙 기반 (0.0, 0.1)"),
  "s2d_twopoint_grid": (Cfg(), f"두 점 반응 (0 → 0.5, 기준 잉여율 → 0.85), 실측 점유율 0.602로 용량 151, T 30, 몫 격자 튜닝 결과 {BEST[6:]}", GRID, BEST),
}

if __name__ == "__main__":
  for tag in sys.argv[1:] or STATES:
    cfg, note, pol, other = STATES[tag]
    print(table(run(cfg, pol, tag, note), "근시안", other), "\n")
