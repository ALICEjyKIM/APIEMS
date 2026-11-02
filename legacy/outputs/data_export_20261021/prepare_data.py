import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from utils.params import Cfg, PROC15H
from env.platform import Env
from bench.rule_split import rule

OUT = Path(__file__).resolve().parent
cfg = replace(Cfg(), **PROC15H)
env, pol = Env(cfg, 0), rule(cfg)
env.reset()
periods, orders, suppliers, deals, pending = [], [], [], [], []
history = json.loads((ROOT / 'src/result/runs/diag_s10_history.json').read_text(encoding='utf-8'))['runs']['Rule'][0]
expected = {kind: {} for kind in ('buy', 'sup')}
for kind in expected:
  for row in history[kind]:
    expected[kind].setdefault(row['t'], []).append(row)
seen_b, seen_s = set(), set()
for t in range(cfg.T):
  o = env.o
  for j, (ready, prof) in sorted(env.pipe.items()):
    for i in np.flatnonzero(prof.qty):
      pending.append([t, j, int(i), int(prof.qty[i]), float(prof.price[i]), ready - t, ready])
  decision = pol.act(o)
  r = env.step(decision)
  for kind, ids, h, e, s, p in [('buy', o.bid, o.hb, r['ok'], r['sb'], r['pb']), ('sup', o.sid, o.hs, r['alloc'], r['ss'], r['ps'])]:
    source = expected[kind][t]
    assert len(source) == len(ids)
    for k, old in enumerate(source):
      np.testing.assert_allclose([h[k], float(e[k]), s[k], p[k]], [old['h'], old['e'], old['s'], old['p']], rtol=1e-9, atol=1e-7)
  for b, bid in enumerate(o.bid):
    for i in np.flatnonzero(o.q[b]):
      orders.append([t, bid, int(i), int(o.q[b, i]), float(o.p[b, i]), int(r['ok'][b]), float(r['sb'][b]), float(r['pb'][b]), int(r['kept_b'][b]), float(o.hb[b]), int(bid not in seen_b)])
  for j, sid in enumerate(o.sid):
    for i in np.flatnonzero(o.cap[j]):
      suppliers.append([t, sid, int(i), int(o.cap[j, i]), float(o.c[j, i]), int(r['alloc'][j]), float(r['ss'][j]), float(r['ps'][j]), int(r['kept_s'][j]), float(o.hs[j]), int(sid not in seen_s)])
  x = decision[0] * r['ok'][:, None, None]
  for b, j, i in zip(*np.nonzero(x)):
    deals.append([t, o.bid[b], o.sid[j], int(i), int(x[b, j, i]), float(o.p[b, i]), float(o.c[j, i])])
  periods.append([t, len(o.bid), sum(b not in seen_b for b in o.bid), len(o.sid), sum(s not in seen_s for s in o.sid), int(r['n_ok']), int(r['stay_buy']), int(r['stay_sup']), float(r['sb'].sum()), float(r['ss'].sum()), float(r['rev']), float(r['ship']), float(r['profit']), int(o.q.sum()), int(o.cap.sum())])
  seen_b.update(o.bid)
  seen_s.update(o.sid)
  assert int(x.sum()) == int(o.q[r['ok']].sum())
  assert np.isclose(r['profit'], r['rev'] - r['ship'])

training_path = ROOT / 'src/result/runs/diag_s11_context_obs.csv'
with training_path.open(encoding='utf-8', newline='') as f:
  reader = csv.DictReader(f)
  keys = reader.fieldnames
  training = [[r[k] if k == 'pol' else float(r[k]) for k in keys] for r in reader]
assert len(training) == 929
labels = ['정책', '반복', '기간', '공급자ID', '유지가치_c', 'c_표준오차', '총공급용량', '공급품목수', '가격수준_평균', '가격수준_최소', '거래이력_hs', '직전재참여확률', '활동기간', '대체업체수_최소', '대체업체수_평균', '대체용량_최소', '대체용량_평균', '공급점유율_최대', '관련품목_총수요', '수요공급비_최대', '수요공급비_평균', '부족이력_최대', '부족이력_평균', '연결주문자수', '연결주문_총수량', '주문자이력_평균', '주문자이력_최소', '묶음최소공급자수_평균', '병목주문수', '대기공급용량', '대기업체수', '최소대기기간', '제외시공급없는품목', '공급자수_최소']
assert len(keys) == len(labels)

def sheet(name, headers, rows, note, source, percent=None, decimal=None):
  return dict(name=name, headers=headers, rows=rows, note=note, source=source, percent=percent or [], decimal=decimal or [])

info = [
  ['데이터 구분', '모형으로 생성한 합성 데이터. 가격·금액은 모형 단위이며 통화는 지정하지 않음.'],
  ['추출 범위', 'PROC15H / Rule / seed=0 / rep=0 / t=0~29의 30기간 재현 실행.'],
  ['기존 기록 대조', '주문자·공급자별 이력, 거래 경험, 배분 잉여, 재참여확률을 기존 s10 기록과 전수 대조.'],
  ['학습표본 범위', '기존에 저장한 공급자 학습 표본 929개. 대표 실행과는 별도의 여러 상태에서 수집.'],
  ['기간별결과', '한 행 = 한 기간. 주문자·공급자 수와 이윤, 주문 성립·잔류를 확인.'],
  ['주문', '한 행 = 기간 × 주문자 × 요구 품목. 품목 수량·가격은 해당 품목 값.'],
  ['공급', '한 행 = 기간 × 공급자 × 공급 품목. 용량·가격은 해당 품목 값.'],
  ['거래배정', '한 행 = 기간 × 주문자 × 공급자 × 품목. 성립한 주문의 실제 배정만 수록.'],
  ['대기공급', '한 행 = 기간 × 공급자 자리 × 품목. 공급자 ID가 아니라 자리 번호로 식별.'],
  ['학습표본', '한 행 = 정책 × 반복 × 기간 × 공급자. c는 Rule 후속 정책 10기간 가치 차이의 40개 미래 평균.'],
  ['중복 합산 주의', '주문·공급의 성립/배정, 잉여, 확률, 잔류, 이력은 참여자 값이 품목마다 반복됨. 잉여 합계는 기간별결과 참조.'],
  ['지표 구분', '재참여확률은 당기 거래 결과로 계산한 확률. 다음기간잔류는 실제 0/1 추첨 결과. 이력은 당기 정산 전 값.'],
  ['새참여자 기준', '처음 관측된 참여자는 1. t=0에는 초기 공급자도 포함.'],
  ['식별자', '기간·품목·참여자·자리 번호는 원 코드와 같이 0부터 시작. 학습표본 결합키에는 정책·반복도 포함.'],
  ['가격', '주문의 제안가격과 공급의 공급가격은 품목당 단가. 거래배정 수량에 곱해 금액 계산 가능.'],
  ['학습 목표 해석', '유지가치는 해당 공급자를 제외했을 때의 Rule 후속 정책 가치 감소 추정치. 최적 가치나 관측 실거래 이윤이 아님.'],
  ['확률·이력 단위', '0~1 사이 값을 백분율로 표시. 수요공급비와 가격수준은 배수로 표시.'],
  ['신규 주문자', '기간당 Poisson(10). 품목 2개를 균등 비복원 추출하고 품목당 수량은 정수 6~15개.'],
  ['시장 규모', '품목 15개, 공급자 자리 24개, 목표 평균 대체 공급자 4개.'],
  ['생성 설정', '공급 편중도 0.5, 품목 자격 이질성 0.5, 공급 여유 1.3, 교체 지연 2기간.'],
  ['이력 설정', '지수평활 계수 0.8, 초기 이력 0.5.'],
  ['원자료', 'diag_s10_history.json / diag_s11_context_obs.csv'],
  ['생성 코드', 'utils/params.py(PROC15H), env/platform.py, bench/rule_split.py'],
]
sheets = [
  sheet('안내', ['항목', '설명'], info, '대표 실행의 세부 원자료와 기존 학습 표본을 함께 수록했습니다.', '모형 코드 및 저장된 진단 원자료'),
  sheet('기간별결과', ['기간', '주문자수', '신규주문자수', '공급자수', '새공급자수', '성립주문수', '주문자잔류수', '공급자잔류수', '주문자잉여합', '공급자잉여합', '배송전플랫폼수입', '배송비', '플랫폼이윤', '요구총수량', '공급총용량'], periods, '한 행 = 한 기간. 금액은 모형 단위. 최초 기간 새공급자수에는 초기 공급자 포함.', 'PROC15H / Rule / seed=0 / rep=0; s10 기존 이력과 대조 완료', decimal=[8,9,10,11,12]),
  sheet('주문', ['기간', '주문자ID', '품목ID', '요구수량', '제안단가', '주문성립_01', '주문자잉여', '재참여확률', '다음기간잔류_01', '성립이력', '새참여자_01'], orders, '한 행 = 주문자 × 품목. 잉여·확률·성립·이력은 품목마다 반복되므로 합산하지 마세요.', 'PROC15H / Rule / seed=0 / rep=0', percent=[7,9], decimal=[4,6]),
  sheet('공급', ['기간', '공급자ID', '품목ID', '공급가능량', '공급단가', '거래배정_01', '공급자잉여', '재참여확률', '다음기간잔류_01', '배정이력', '새참여자_01'], suppliers, '한 행 = 공급자 × 품목. 잉여·확률·배정·이력은 품목마다 반복되므로 합산하지 마세요.', 'PROC15H / Rule / seed=0 / rep=0', percent=[7,9], decimal=[4,6]),
  sheet('거래배정', ['기간', '주문자ID', '공급자ID', '품목ID', '배정수량', '주문제안단가', '공급단가'], deals, '성립한 주문의 배정만 수록. 주문·공급 시트와 기간 및 참여자ID, 품목ID로 연결됩니다.', 'PROC15H / Rule / seed=0 / rep=0', decimal=[5,6]),
  sheet('대기공급', ['기간', '공급자자리', '품목ID', '후보공급용량', '후보공급단가', '남은대기기간', '활동예정기간'], pending, '거래 전 관측 시점의 대기 후보. 공급자자리는 참여자 ID와 다릅니다.', 'PROC15H / Rule / seed=0 / rep=0', decimal=[4]),
  sheet('학습표본', [f'{a}\n({b})' for a,b in zip(labels, keys)], training, '저장된 929개 표본 전체. c는 유지가치 추정치이며 se는 Monte Carlo 표준오차입니다.', 'src/result/runs/diag_s11_context_obs.csv', percent=[10,11,17,21,22,25,26])
]
payload = dict(sheets=sheets, counts={s['name']:len(s['rows']) for s in sheets}, profit=sum(r[12] for r in periods), source_history_matched=True)
(OUT / 'export_data.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
print(json.dumps({k:v for k,v in payload.items() if k != 'sheets'}, ensure_ascii=False))
