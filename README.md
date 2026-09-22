# 5PL 미들마일 플랫폼의 동적 매칭 및 잉여배분 (APIEMS 2026)

다품종 묶음 주문을 내는 주문자와 품목별 공급 용량을 가진 공급자를 매 기간 매칭하고, 거래잉여를 주문자·공급자·플랫폼에 나누는 플랫폼을 다룬다.
참여자가 다음 기간에 다시 올 확률은 이번 기간에 받은 잉여에 달려 있다. 따라서 오늘의 배분이 내일의 참여자 구성을 바꾼다.

**연구 질문**: 미래 가치를 어떻게 근사해야 이런 플랫폼의 동적 의사결정이 좋아지는가?

**방법**
1. 시뮬레이션 경험으로 미래 가치 V(s)를 학습한다(선형 / MLP / GNN).
2. 참여자별 유지 가치 c_i = V(s) − V(s에서 i 제외)를 구한다.
3. 매 기간 MILP(당기 이윤 + Σ c_i × 재참여확률_i)로 주문 수락, 공급자·수량 배정, 잉여배분을 한 번에 정한다.

> 소규모 예비 실험이다: 품목 6, 공급자 자리 6, 기간 30, 정책 비교 30반복.

## 현재 상태 (2026-09-22)

| 단계 | 상태 |
|---|---|
| 시장 환경, 비교군(근시안·규칙 기반), 선형 근사 | 완료 |
| 실험 1: 선형 vs MLP | 완료 |
| 실험 3 v1: MLP vs MLP+편중도 요약 vs GNN (축소 4칸, 예측·변환 오차만) | 완료 |
| 실험 3 v1 진단 | 완료 |
| 실험 3 v2 (진단 처방 적용, 정책 평가) | 다음 |
| 실험 2: 반응 함수 강건성 | 보류 |

지금까지의 결과:
- **실험 1**: 가설과 달리 MLP가 선형 근사보다 낫지 않았다. 두 가치 근사 정책은 규칙 기반보다 나았지만 근시안은 넘지 못했다.
- **실험 3 v1**: GNN이 우위라는 근거를 찾지 못했다. 다만 진단 결과 학습 정답의 잡음이 크고(달성 가능한 R² 상한 약 0.46) GNN 입력에 척도 결함이 있어, 공정한 비교는 아직 이루어지지 않았다.

자세한 결과와 해석은 [진행 문서](docs/tex/apiems_progress.tex)에 있다.

## 설치

Python 3.11 conda 환경과 **Gurobi 라이선스**(학술 라이선스 가능)가 필요하다. MILP는 모두 gurobipy로 푼다.

```bash
conda create -n apiems python=3.11
conda activate apiems
pip install -r requirements.txt
```

## 실행

모든 명령은 `src/`에서 실행한다.

```bash
cd src
python -m pytest tests -q -W error          # 테스트 145개, 약 1분
```

실험 스크립트는 단계 이름을 인자로 받는다. 결과는 `src/result/runs/*.json`에만 저장된다.

| 명령 | 내용 | 산출물 | 시간 (참고) |
|---|---|---|---|
| `python -m result.exp1_linear_vs_nn value` | 실험 1 학습, 예측·변환 오차 (튜닝 seed) | `exp1_value.json` | |
| `python -m result.exp1_linear_vs_nn eval` | 실험 1 정책 비교 (평가 seed, **한 번만**) | `exp1_eval.json` | |
| `python -m result.exp3_graph mlp` | 실험 3 MLP 두 종과 시뮬레이션 기준치, 4칸 병렬 | `exp3_mlp.json` | 약 55분 |
| `python -m result.exp3_graph gnn` | 실험 3 GNN, 4칸 병렬 | `exp3_gnn.json` | 약 23분 |
| `python -m result.exp3_graph table` | 실험 3 표 (json에서) | 화면 출력 | 수 초 |
| `python -m result.exp3_diag run fig table` | 실험 3 v1 진단: 학습 곡선, 정답 잡음, GNN 점검, 모델 저장 | `exp3_diag.json`, `result/figures/`, `result/models/` | 약 16분 |
| `python -m result.plots all` | 진행 문서용 그림 (json에서) | `result/figures/*.png` | 수 초 |
| `python -m result.diag_slice2`, `diag_slice3` | 슬라이스 2·3 비교군 진단 재현 | `diag_*.json` | |

시간은 24코어 PC에서 실제로 잰 값이다(빈칸은 기록 없음). 한 번의 실행이 무거운 경우가 있으므로 백그라운드 실행을 권한다.

## 폴더 구조

```
src/
  utils/    params.py(모든 수치 설정, frozen Cfg), common(난수 스트림), arrivals, instance, pwl(구간선형 꺾인 점), features(시장 요약 지표)
  env/      platform.py(환경·정산·재참여), graph.py(거래 연결 그래프, 편중도 요약), response.py
  bench/    myopic.py, rule_split.py(규칙 기반 몫과 튜닝), tune.py(반응 보정)
  match/    milp_build.py·milp_solve.py(모든 정책이 공유하는 MILP), interface.py(rollout), mc_value.py(시뮬레이션 기준치)
  vfa/      linear.py, mlp.py, gnn.py, train.py(경험 수집·설정 선택), coef.py(유지 가치 변환)
  result/   실험·진단 스크립트, runs/(원자료 json), figures/, models/(저장한 학습 모델)
  tests/
docs/tex/   apiems_progress.tex(진행 문서, figures/ 포함), facts.md(tex 갱신용 사실 정리)
ref/plan/   roadmap.md(슬라이스 계획과 모든 결정 기록)
CLAUDE.md   연구 설계, 모델 가정, 코딩 컨벤션, 공정성 원칙 (기준 문서)
```

`match/hindsight.py`, `result/exp2_response.py`, `result/summary.py`는 아직 비어 있다.

## 진행 문서 컴파일

`docs/tex/` 폴더를 통째로 Overleaf에 올리고 Compiler를 **XeLaTeX**로 설정한다.
폰트는 Noto Serif/Sans CJK KR과 DejaVu Sans Mono를 쓴다.

## 연구 규칙 (요약)

전체 규칙은 [CLAUDE.md](CLAUDE.md)에 있다.

- **공통 설정**: 모든 수치는 `utils/params.py`의 `Cfg`에만 둔다. 모든 정책은 같은 MILP를 쓰고 유지 가치만 다르다.
- **공통 난수**: 정책 비교는 공통 난수로 한다. 튜닝·학습은 튜닝 seed(1)로 하고, 평가 seed(0)는 최종 비교에 한 번만 쓴다.
- **원자료**: 모든 표와 그림은 `result/runs/*.json` 원자료에서만 만들고, 출처 파일을 적는다.
- **설정 고정과 기록**: 실험 설정은 결과를 보기 전에 [roadmap.md](ref/plan/roadmap.md) 결정 기록에 고정한다. 결과를 본 뒤 바꾼 것은 이유와 함께 기록한다.
- **공정한 비교**: 비교군을 약하게 만들지 않고, 모든 근사 방법에 같은 데이터와 같은 튜닝 예산을 준다.
- **불리한 결과**: 가설과 반대인 결과도 숨기지 않고 기록한다.
