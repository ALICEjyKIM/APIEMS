"""공통 난수 스트림.
난수는 (seed, 반복, 스트림, 키)로 생성하므로 정책이 달라도 같은 참여자·기간에는 같은 난수가 나온다.
스트림: inst(인스턴스), arr(신규 도착), noise(재참여 변동), ret(재참여 판정), mc(MC 가치).
"""
import numpy as np

STREAMS = {"inst": 0, "arr": 1, "noise": 2, "ret": 3, "mc": 4}
KINDS = {"buy": 0, "sup": 1}


# (seed, rep, stream, keys)로 고정된 난수 생성기
def rng(cfg, rep, stream, *keys):
  return np.random.default_rng([cfg.seed, rep, STREAMS[stream], *keys])


# 참여자·기간별 재참여 판정용 균등난수 (u < p이면 재참여)
def ret_u(cfg, rep, kind, pid, t):
  return rng(cfg, rep, "ret", KINDS[kind], pid, t).random()
