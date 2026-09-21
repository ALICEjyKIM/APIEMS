"""슬라이스 2 진단 상태의 원자료 저장.
환경 변경 전후 세 상태(용량 91·T 25, 용량 182·T 30, 충원 확률 0.25)를 같은 코드로 다시 돌려 반복별 원자료를 남긴다.
공통 난수와 고정 솔버 seed 덕에 당시 표가 그대로 복원된다. 실행: src/에서 python -m result.diag_slice2 [상태 태그...].
"""
import sys
from dataclasses import replace
from utils.params import Cfg
from match.milp_solve import Policy
from result.diag import run, table

RULES = {f"규칙 기반 (0.3, {s})": (lambda c, s=s: Policy(c, split=(0.3, s))) for s in (0.1, 0.2, 0.3, 0.4)}
POLICIES = {"근시안": lambda c: Policy(c), **RULES}
# cover 0.65는 점유율 나눗셈 도입 전의 용량 91(= 1.3 × 70)을 현재 공식(cover × 70 ÷ 점유율 0.5)으로 재현하기 위한 값
STATES = {
  "s2a_cap91_T25": (replace(Cfg(), T=25, p_sup_new=0.5, cover=0.65, b_buy=39.55078125, b_sup=24.70703125),
                    "슬라이스 2 종료 시점: 용량 91, T 25, p_sup_new 0.5, 옛 기준 보정 (39.55, 24.71)"),
  "s2b_cap182_T30": (replace(Cfg(), T=30, p_sup_new=0.5, b_buy=31.73828125, b_sup=35.25390625),
                     "환경 변경 (B)(D) 후: 용량 182, T 30, p_sup_new 0.5, 옛 기준 보정 (31.74, 35.25)"),
}

if __name__ == "__main__":
  for tag in sys.argv[1:] or STATES:
    cfg, note = STATES[tag]
    print(table(run(cfg, POLICIES, tag, note), "근시안", "규칙 기반 (0.3, 0.3)"), "\n")
