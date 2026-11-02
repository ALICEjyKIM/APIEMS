"""근시안 정책 (참고용 하한).
유지 가치 0: 매 기간 플랫폼 이윤만 최대화하므로 잉여를 주지 않고 참여자 대부분이 떠난다.
미래를 전혀 고려하지 않을 때의 참고값이며, 가치 근사 정책의 주 비교 기준은 규칙 기반이다.
"""
from match.milp_solve import Policy


# 근시안 정책: 같은 MILP, 유지 가치 0, 잉여 자유
def myopic(cfg):
  return Policy(cfg)
