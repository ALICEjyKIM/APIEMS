# 내생적 참여자 구성과 학습 가치함수의 MILP 임베딩: 1차 문헌 검토 (인용 검증판)

조사일: 2026-09-28 · 검증일: 2026-09-28 · slug `endogenous-participation-vfa-embedding`
범위: `ref/literature-review.md` 1차(주제 A·B). C(GNN VFA 한계) · D(묶음 초모듈성) · E(배분 몫 특성화)는 2차로 미룬다.
원자료: `ref/_review/..._research_A.md`(증거 34건, 원 번호 A[1]–A[34]), `..._research_B.md`(증거 28건, 원 번호 B[1]–B[28]).
이 검증판에서는 두 원자료의 번호를 **단일 연속 번호 [1]–[42]**로 통합했다(실제 본문에서 인용된 항목만 포함,
원자료에는 있으나 본문에서 인용되지 않은 항목은 목록에서 제외). 원 번호 대응표는 문서 끝 `## 부록: 원 번호 대응표`에 남긴다.
모든 URL은 이 검증판 작성 시 WebFetch로 재확인했다(§8. 통합 Sources의 상태 표기 참고). "검증됨"·"확인됨"·"재현됨"이라는
표현은 실제로 원문을 대조했거나 저장소 코드/원자료 수치를 대조한 경우에만 쓴다.

이 문서의 목적은 문헌 요약이 아니라 **갭 특정**이다. 결론부터: 우리가 당초 방법론 기여로 지목했던
"학습된 가치함수를 MILP가 직접 최대화한다"는 **선례가 4편 있어 기여가 아니다.** 기여는 다른 곳으로
재배치해야 하고, 재배치할 자리는 있다.

---

## 1. 조사 방법과 한계

WebSearch로 지형을 잡고 **OpenAlex API 인용 그래프 순회**(참조 목록·피인용·공통 인용자 계수)를 주 수단으로 썼다.
키워드 검색만으로는 이 주제가 잡히지 않는다 — 조사 착수 시 시험 쿼리의 1위 결과가 무관한 서베이(*Knowledge Graphs*,
인용 1835)였다. 씨앗 논문의 참조·피인용을 교차해 **여러 씨앗이 공통으로 참조하는 논문**과
**두 군집의 공통 인용자가 0인 지점**을 갭 신호로 삼았다.

한계를 먼저 밝힌다.

- **저널 사분위를 직접 확인한 항목이 없다.** 원자료에서 전부 "사분위 미확인"으로 표기했다.
- **전문을 읽은 논문은 소수다.** 주제 A는 3편(Luy 외 arXiv판 [3], Chen·Xu arXiv [14], Balseiro 외 NeurIPS PDF [11]),
  주제 B는 4편 남짓(van Heeswijk ar5iv [18], Delarue 외 [19], van Steenbergen 외 해당 절 [20], Iyer 외 정의식 [27]).
  나머지는 초록 또는 서지 수준이며 통합 Sources(§8)에 항목마다 읽은 수준을 표기했다.
  다만 [35](Zhang 외, GNN 최적화)는 **초록과 인용 문장만** 확인했다. §3.3에서 이 논문의 사정거리를 논하는 부분은
  전문을 근거로 한 것이 아니라 우리의 해석임을 명시한다.
- **INFORMS·Wiley·Elsevier·SSRN은 봇 차단으로 접근하지 못했다.** 특히 Chen 외 (2020) TR-B는 검색 요약 경유
  medium confidence이고, Hildebrandt 외 (Networks 2026)과 Delarue·Lian·Qin (2026 SSRN)은 초록조차 확보하지 못했다.
  이 두 편은 각각 B2와 A2의 판정을 바꿀 수 있다.
- OpenAlex 색인 한계가 있다(INFORMS 초록 미수록이 많다). "0건"은 "존재하지 않음"이 아니라
  "이 색인과 이 쿼리로는 걸리지 않음"이다. 원자료에 0건 쿼리를 전부 남겼다.

---

## 2. 주제 A: 내생적 참여자 구성

### 2.1 세 계보로 갈라져 있고, 서로의 방법을 쓰지 않는다

해당 연구는 세 계보 중 하나에 속한다 (A 원자료 §4.A1).

| 계보 | 대표 | 참여 내생 | 기간 간 상태 전이 | 금전 배분 결정 |
|---|---|---|---|---|
| 인력계획 MDP | Gans·Zhou 2002 [1] → Arlotto 외 2014 [2] → Luy 외 2024 [3] | O | O | X |
| 플랫폼 경제학 | Taylor 2018 [4], Cachon 외 2017 [5], Bhargava 외 2022 [6] | O | **X (정태 균형)** | O |
| 동적 매칭 | Hu·Zhou 2022 [7], Aouad·Sarıtaç 2022 [8], You·Vossen 2024 [9] | 부분(**외생 이탈**) | O | X |

이 전이를 부르는 방법론 라벨은 **decision-dependent (endogenous) uncertainty**다 (Hellemo 외 2018, [10]).

### 2.2 A2 직답: 세 갈래로 나뉘어 있고, 우리 조합은 비어 있다

1. **잉여배분 몫을 기간별 결정변수로 둔 동적 모형은 있다** — Balseiro 외 (NeurIPS 2017) [11].
   판매자별 몫을 기간마다 정한다. **그러나 참여가 기대값 IR 제약이고 참여자 집합이 변하지 않는다.**
2. **수수료율을 기간별 결정변수로 두고 ADP로 푼 모형도 있다** — Chen, Zheng, Ke, Yang (2020) *TR-B* 138:23–45, [12].
   전략이 도착률에 영향을 준다. **그러나 수수료율이 시장 전체 단일 스칼라이고 반응이 집계 도착률**이어서
   "참여자 i를 잃는 손해 c_i"라는 개념이 성립하지 않는다.
3. **참여자 이탈이 결정에 내생인 ADP도 있다** — Luy, Hiermann, Schiffer (2024) *POM*, [3].
   **그러나 결정변수가 고정기사 채용 수(스칼라)이고, 이탈이 금전이 아니라 미매칭 비율에 반응한다.**

빈 칸: 참여자 **개별**에게 가는 잉여 몫을 기간별 결정변수로 두고, 그 몫이 개별 **재참여확률**을 통해
**다음 기간 참여자 집합과 거래 연결 구조**를 바꾸며, 이를 **하나의 매칭 MILP**로 함께 결정하는 모형.
(1)·(2)는 배분을 결정하지만 집합을 바꾸지 않고, (3)은 집합을 바꾸지만 배분을 결정하지 않는다.

### 2.3 Luy 외 (2024) POM — 가장 위험하고 동시에 가장 유용하다

구조적으로 가장 가까운 선행연구이므로 **반드시 인용·차별화해야 한다**. arXiv 본문에서 확인한 사실:

- 이탈확률은 미매칭 비율에 대한 두 점 선형 혼합이고(p_high = 1, p_low = 0.01), 이탈 인원은 이항분포다.
- VFA는 **고정기사 차원의 분리형 구간선형(PL-VFA), 오목성 유지** — 본문 3.3.2절에
  "We seek for a piecewise linear approximation of V along the FD dimension"으로 적혀 있다.
- 비교군은 PL-VFA vs 근시안 채용정책, 그리고 초록이 언급하는 **완전정보 lookahead**다.

세 가지 의미가 있다.

1. 우리가 "기존 ADP의 표준 근사구조"로 지목한 **분리형 구간선형이 2024년 POM 논문의 실제 선택**이다.
   "분리형 → 비분리형"이라는 우리 기여의 축이 최신 문헌을 상대로 성립한다.
2. 반응함수를 **두 점으로 고정**하는 방식이 우리(잉여 0 → 0.5, 기준 잉여율 → 0.85)와 유사하다.
   우리 두 점 고정에 대한 방어 근거로 쓸 수 있다.
3. 완전정보 lookahead 비교군이 이 문헌의 표준이다 (§2.4).

### 2.4 A5: 우리 비교군에 상한이 없는 것이 문헌 기준 미달이다

문헌 표준은 정책 간 비교에 그치지 않는다.

- **LP/ALP 완화 상한과 최적성 격차**: You·Vossen (2024) [9]은 ALP 재정식화로 실행 가능 정책과 상한을
  동시에 얻어 격차를 보고한다. Aouad·Sarıtaç (2022, *Operations Research*) [8]은 LP 벤치마크와
  상수비 근사 보증을 제시한다.
- **완전정보(후견) 상한**: Luy 외 [3]이 비교군으로 쓴다. ADP 일반의 표준 어휘다.

우리 비교군은 규칙 기반·근시안·선형 VFA로 **전부 정책**이다. `src/match/hindsight.py`는 실제로 **0바이트로 비어
있다**(코드 확인). 상한 대비 격차를 결과 표에 싣지 않으면 문헌 기준에 못 미친다. 이는 선택 사항이 아니다.

반대로 우리가 **더 보수적인 점**도 분명히 있다. [3]·[12]는 근시안 대비 개선이 주된 근거인데,
우리는 근시안을 참고용 하한으로 내리고 4×4 격자로 튜닝한 규칙 기반을 주 비교 기준으로 삼는다.

### 2.5 A3·A4: 두 개의 부수적 공백

- **A3 반응함수**: 관찰된 함수형은 유보값·임계형([4][5][13]), 두 점 선형 혼합([3]), 오목·포화형([14]),
  집계 도착률([12]), 데이터 추정 생존모형([15][16])이다.
  **로지스틱을 쓴 선례는 확인하지 못했고, 반응함수 오지정 강건성을 플랫폼 유지 맥락에서 다룬 연구도 없다.**
  보류해 둔 실험 2가 놓이는 공백이다. 반박 근거는 Masorgo 외 (2026 *JOM*) [17] — 기사 반응이 금전만의
  함수가 아니라는 실증(서지만 확인).
- **A4 응용 맥락**: **없음.** 5PL 문헌(OpenAlex 72건)은 개념·역량 논의이고 최적화 모형이 아니다.
  `middle-mile transportation platform matching` 2건(무관), `middle mile consolidation hub dynamic optimization
  platform` 0건, `all-or-nothing bundle order platform dynamic allocation supplier` 0건,
  `B2B platform dynamic matching suppliers buyers` 0건. 인접 문헌(협력 운송 이익배분)은 Shapley 기반 정태 협조게임이다.

---

## 3. 주제 B: 학습 가치함수의 MILP 임베딩

### 3.1 B2·B3 직답: 선례가 있다. 갭 주장을 좁혀야 한다

학습된 **비분리형** 신경망 VFA를 매 기간 MILP에 big-M으로 정확 임베딩하고 ADP로 반복 해결한 연구가 4편이다.

| 연구 | 게재 | 임베딩 | 규모·해 시간 | 읽은 수준 |
|---|---|---|---|---|
| van Heeswijk·La Poutré 2019 [18] | arXiv (게재 미확인) | ReLU big-M, 정수 MILP, ADP 사후상태 VFA | 은닉 1×20 / 3×20 → 반복당 0.16 / 0.39초 | 전문(ar5iv) |
| Delarue, Anderson, Tjandraatmadja 2020 [19] | NeurIPS | 가치함수 MIP 임베딩 + 정책 반복 | 은닉 1층 16뉴런, Gurobi 0.4초(입력 21) / 39초(입력 51) | 전문 |
| van Steenbergen, van Heeswijk, Mes 2025 [20] | **Transportation Science** 59(2):360–390 | NN-VFA를 big-M으로 MIP에, **분리형 DL-VFA와 직접 비교** | 0.03–0.20초/의사결정 에폭 | 전문(해당 절) |
| Hildebrandt, Bode, Ulmer, Mattfeld 2026 [21] | **Networks** | 초록: 벨만 방정식을 MILP로, 가치함수는 신경망 | 미확인 | **페이월 403** |

따라서 **"학습된 V를 사후상태 가치로 두고 MILP가 직접 최대화한다"는 것 자체는 우리 기여가 아니다.**
특히 [20]은 비분리형 NN-VFA를 분리형 VFA와 직접 비교한 Transportation Science 논문이므로,
"분리형 vs 비분리형 비교"라는 프레이밍도 단독으로는 기여가 되지 않는다.

도구 계층(OptiCL/Maragno 외 *Operations Research* 2025 [22], JANOS [23], OMLT [24],
gurobi-machinelearning [25])과 이상적 정식화(Anderson 외, *Mathematical Programming* 2020 [26])는 이미 표준이다.

### 3.2 B4 직답: 우리 c_i의 기존 용어가 존재하고, 편향도 이미 지적되어 있다

`c_i = V(S) − V(S∖{i})`는 세 문헌군에 이미 있다.

1. **집합함수 이론이 가장 정확하다.** Iyer–Jegelka–Bilmes (ICML 2013) [27]의
   **discrete semigradient(supergradient)** 성분과 정확히 같다. 그 논문은 이것이 정의하는 모듈러(선형)
   한계가 **현재 해에서 tight**하다고 적는다. 즉 참여자 일부가 떠난 집합에서 근사가 어긋나는 것은
   **정의상 예견된 바**이며, 발견이 아니다.
2. **ADP·확률계획**: 좌표 방향 표본 기울기로 분리형 구간선형 근사를 만드는 것이 표준이다 (SPAR [28], 튜토리얼 [29]).
3. **다중 에이전트 RL**: value decomposition(VDN [30]), difference rewards(COMA [31]).

편향 지적도 이미 있다.

- SPAR (Powell–Ruszczyński–Topaloglu, *Mathematics of Operations Research* 2004) [28]은 분리형 근사의
  오차 상한을 주고 **주된 오차원이 분리형 근사의 사용**이라고 적는다.
- Powell–Topaloglu 튜토리얼 [29]: **자원 종류가 여럿일 때 단순한 분리형 VFA는 잘 작동하지 않는다.**
  우리의 "여러 품목 × 여러 공급자"가 이 다중 자원 종류에 구조적으로 대응한다(추론).
- NeurADP (AAAI 2020) [32]는 우리와 **구조가 같은 선례**다. 결합 가치를 차량별 항의 합으로 분해하고
  한 에폭 안에서 다른 차량의 행동이 이 차량의 장기가치를 크게 바꾸지 않는다고 가정을 명시한다.
  **다만 확인한 범위에서 그 분해 오차를 측정·보고하지 않는다.**

**반대 방향 증거도 분명히 있다.** Topaloglu–Powell (*IJOC* 2006) [33]은 대체가 있으면 정확한 가치함수가
비분리형이라고 인정하면서도, 분리형 구간선형 근사가 **모든 대체 패턴에서 고품질 해**를 준다고 보고한다.
즉 "비분리성이 있다 → 분리형 근사가 나쁘다"는 자동으로 성립하지 않는다.

**우리 실험 1 결과가 이 방향과 일치한다(코드·원자료로 확인).** `src/result/runs/exp1_value.json`·`exp1_eval.json`을
직접 대조한 수치는 다음과 같다: 검증 분할 R²는 선형 0.3468, MLP 0.2083으로 선형이 오히려 높고, 절대 예측오차의
짝지은 차이(선형 − MLP)는 평균 −35.04(95% CI 반폭 333.23, 상태 24개)로 0을 포함해 **두 방법이 통계적으로
구분되지 않는다**. 정책 평가에서도 MLP 유지 가치 − 선형 유지 가치의 누적 이윤 차이는 평균 −810.01(95% CI 반폭
1011.62, 30반복)로 역시 구분되지 않으며, 두 근사 모두 근시안을 유의하게 넘지 못했다(근거: `docs/tex/facts.md`
§6 `exp1_value.json`·`exp1_eval.json` 절, 결정 기록 `roadmap.md:461-484`). 즉 "선형 ≈ MLP, 분리형으로 충분하다"는
관측(정확히는 "차이를 통계적으로 가려낼 수 없었다")은 문헌과 모순되지 않고 **[33] 쪽 증거**로 읽을 수 있다.
[29]의 "다중 자원 종류에서 분리형은 잘 안 된다"와는 긴장 관계에 있다.
그 긴장이 해소되는 조건(공급 편중도, 묶음 크기)을 재는 것이 실험 3의 위치다. 이 프레이밍이 정직하고,
우리 음의 결과를 버리지 않는다.

### 3.3 B5: 대안 설계에 유리한 증거와 불리한 증거가 모두 있다

- **유리**: Graph4BiLO (2026 프리프린트) [34]은 GNN 전체를 임베딩하면 ReLU 활성 약 28,896개(노드 100),
  노드 40에서 1049초, **노드 60 이상에서 1시간 타임아웃**을 보고한다. **헤드만 임베딩하는 설계가 옳은 방향이다.**
  선형 읽기(readout) 자체도 [34]에 선례가 있다. 다만 **풀링 가중치가 결정변수인 구조는 찾지 못했다.**
- **불리 — 반드시 선제 대응해야 한다**: Zhang 외 (NeurIPS 2023) [35]은 고전적 GNN 구조에서 **그래프가
  고정되면 GNN에 대한 최적화가 밀집 신경망에 대한 최적화와 동등**하다고 명시한다. 그래서 그들은 각 간선이
  결정변수인 경우만 연구한다.
- 가장 가까운 것은 Okada 외 (2026) [36]: 사후결정 잔여 그래프를 GNN으로 근사하지만 이탈이 **외생**이고
  GNN을 최적화에 넣지 않고 **forward-greedy 휴리스틱**으로 대체한다.

**[35]의 사정거리는 우리가 좁혀서 읽은 것이다 — 원문 전체가 아니라 초록과 인용 문장만 확인했다는 점을 밝힌다.**
확인한 범위에서 [35]이 명시하는 것은 "고전적 GNN 구조에서 그래프가 고정되면 GNN에 대한 최적화가 밀집
신경망에 대한 최적화와 동등하다"는 진술뿐이다. 이 진술을 *훈련된 GNN 위에서의 최적화 정식화*에 관한 것으로 읽고,
*예측기로서 GNN의 가치*(파라미터 공유, 순열 불변성, 가변 참여자 수 처리, 거래 연결 구조의 귀납적 편향)와는
별개라고 구분하는 것은 **저자들이 직접 그렇게 프레이밍한 것이 아니라 우리의 해석이다.** 이 해석이 맞다면
[35]은 "GNN이 필요 없다"는 반박이 아니라 **GNN 전용 MILP 정식화를 만들지 말고 헤드만 임베딩하라는 설계
근거**로 읽을 수 있다. 우리 대안 설계가 정확히 그 형태다. 논문 본문에서는 이 구분이 우리 해석임을 먼저
밝히고, 여유가 되면 전문을 확인해 이 읽기가 저자 의도와 어긋나지 않는지 점검해야 한다.

### 3.4 계산 가능성

우리 규모(품목 6, 공급자 6, 주문자 약 30)에서 MILP에 들어가는 것은 **헤드만**이다. 은닉 폭 16, 1–2층이면
ReLU 이진변수 16–32개로, [19](16뉴런 0.4초), [20](0.03–0.20초/에폭), [18](3×20에서 0.39초)의
보고 범위에 **명확히 들어간다.** 기대 그래프 표현 z = Σ p_i h_i는 p_i가 [0,1]이고 h_i가 상수이므로
z의 상·하한을 **해석적으로 정확히** 줄 수 있어 big-M 조임에 유리하다(추론, 미구현).

**"주문자 약 30" 근거를 밝힌다.** 품목 6·공급자 6은 `src/utils/params.py`의 `Cfg`에 고정값(`n_items=6`,
`n_sup=6`)으로 박혀 있지만, 주문자 수는 고정값이 아니라 기간당 평균 `lam_buy=10`의 포아송 신규 도착과
참여자별 재참여확률이 누적된 결과다. 실제 관측된 "활동 주문자" 평균은 정책·기간에 따라 약 11~29 사이이고
(예: `diag_s2a_cap91_T25.json`에서 근시안 11.0·규칙 기반 20.2, `exp1_eval.json`에서 근시안 19.5·규칙 기반 29.05 —
`docs/tex/facts.md` §6), "약 30"은 이 관측 범위의 상단(규칙 기반 정책, 평가 seed)을 딴 근사치다. 따라서 배정
정수변수 "약 1,080개"(= 주문자 수 × 공급자 6 × 품목 6, `src/match/milp_build.py`의 `x = m.addMVar((len(o.q),
len(o.cap), o.q.shape[1]), ...)`)와 노드 수 "36개"(= 주문자 약 30 + 공급자 6)도 고정 상수가 아니라 이 근사치에
연동된 근사 수치로 읽어야 한다.

두 가지 주의. (1) 우리 MILP에는 배정 정수변수 약 1,080개와 참여자별 PWL이 이미 있으므로 결합 후 해 시간은
위 수치보다 커진다. **결합 규모의 해 시간을 보고한 문헌은 찾지 못했다.** (2) GNN 전체 임베딩은 위험하다 —
우리 노드 36개에 은닉 32를 곱하면 층당 활성이 1,000개를 넘어 [34]의 노드 40(1049초) 근방으로 들어간다(추론).

### 3.5 B6: 학습 목적함수의 실패 양식과 우리 가드

1. **외삽·신뢰 불가**: 신뢰영역이 없으면 최적화해에서의 예측모델 평가를 신뢰할 수 없다 [37].
   OptiCL은 관측 데이터의 볼록껍질로 신뢰영역을 잡고 [22], Shi 외 (*IJOC* 2024) [37]는 isolation forest를 쓴다.
2. **optimizer's curse** (Smith–Winkler, *Management Science* 2006) [38]: 추정치가 불편이어도
   최대화 선택 때문에 선택된 대안의 실제 가치가 추정치보다 작다.
3. **실제 가드 사례**: Delarue 외 [19]는 학습된 가치를 알려진 하한으로 감싸 쓰고 성능 개선을 보고한다.
   **신뢰영역이나 불확실성 정량화는 다루지 않는다.**

우리 `coefs()`의 가드(종류별 평균 shrink 후 절단)는 [19]의 하한 감싸기에 가까운 **사후 절단**이며
입력공간 신뢰영역이 아니다. 공급자 c의 체계적 과소평가와 가드의 상호작용을 점검해야 한다.

---

## 4. 통합 갭 교차표

O = 있음, X = 없음. 행은 각 축의 최근접 선행연구다.

| 연구 | 다기간 동적 | 참여자 집합 내생 | 배분이 결정변수 | 개별 배분 | 다품목 묶음 | 학습 VFA | 매칭을 IP로 | 비분리형 VFA를 IP에 임베딩 | 분해 오차 정량화 |
|---|---|---|---|---|---|---|---|---|---|
| Luy 외 2024 POM [3] | O | O | X | X | X | X (분리형 PL) | X | X | X |
| Chen 외 2020 TR-B [12] | O | O (도착률) | O (스칼라) | X | X | X | X | X | X |
| Balseiro 외 2017 [11] | O | X (IR 제약) | O | O | X | X | X | X | X |
| Chen·Xu 2026 [14] | O | O | X (수량) | O | X | X | X | X | X |
| You·Vossen 2024 [9] | O | X | X | X | X | X (ALP) | O | X | X |
| van Steenbergen 외 2025 TS [20] | O | X | X | X | X | O | O | **O** | X |
| Delarue 외 2020 [19] | O | X | X | X | X | O | O | **O** | X |
| NeurADP 2020 [32] | O | X | X | X | X | O | 부분 | X (개체별 분해) | X |
| Okada 외 2026 [36] | O | X (외생) | X | X | X | O (GNN) | X (greedy) | X | X |
| Iyer 외 2013 [27] | — | — | — | — | — | — | — | — | 형식적 tight성만 |
| **우리 설정** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | 목표 | 목표 |

두 개의 서로 다른 단절이 **같은 구조**로 나타난다.

- **A쪽 단절**: Luy 외의 참조 목록에 수익배분 문헌이 한 편도 없고, 역으로 수익배분 문헌은 ADP를 쓰지 않는다.
  동적 최적화 씨앗 4편의 공통 참조는 Powell ADP 교과서와 Cachon 외 2017 단 2편이다.
  플랫폼·매칭 씨앗 8편의 공통 핵은 전부 정태 가격·임금 균형 모형이다.
- **B쪽 단절**: 제약학습(OptiCL/JANOS/OMLT/Maragno) 군집과 ADP 분리형 근사(Powell/Topaloglu) 군집의
  **공통 인용자가 검사한 9개 조합 전부 0건**이다. Maragno 외 인용자 60건을 dynamic·multiperiod·value
  function으로 좁히면 0건 — OCL 문헌은 아직 다기간 동적으로 확장되지 않았다. 두 군집을 잇는 유일한 경로는
  Anderson 외 2020 **정식화** → Delarue 외 2020 → van Steenbergen 외 2025다.
- 집합함수 semigradient 문헌 [27]은 **어느 군집과도 공통 인용자가 0**이다. 노드 삭제 차분의 편향을
  형식적으로 해석하는 어휘가 ADP·VFA 문헌에 아직 수입되지 않았다.

---

## 5. 갭 후보와 반박 대응

### 살아 있는 갭 (우선순위 순)

**G1. 문제 설정 (가장 안전하다).** 참여자 개별 잉여 몫이 기간별 결정변수이고, 그것이 개별 재참여확률을 통해
다음 기간 참여자 집합과 거래 연결 구조를 바꾸며, 이를 하나의 매칭 MILP로 함께 결정하는 모형.
A2의 세 갈래([11] 배분·집합 불변 / [12] 스칼라·집계 반응 / [3] 집합 전이·배분 없음)가 모두 비켜 간다.
다품목 all-or-nothing 묶음과 5PL 미들마일 맥락은 A4에서 공백으로 확인됐다.
**위험**: Delarue·Lian·Qin (2026 SSRN)을 확인하지 못했다.

**G2. 결정 의존 풀링(readout).** z = Σ p_i h_i에서 h_i는 결정 무관 GNN 임베딩, p_i는 MILP 안의
PWL 재참여확률이므로 z가 결정변수의 선형식이고, 작은 헤드만 정확 임베딩한다.
선형 읽기 자체는 [34]에 선례가 있으나 **풀링 가중치가 결정변수인 구조는 찾지 못했다.**
[35]에 따르면 그래프가 고정된 우리 설정에서 이 설계는 필요한 만큼만 임베딩하는 옳은 선택이고,
[34]의 타임아웃이 GNN 전체 임베딩의 위험을 보여준다.
**위험**: Hildebrandt 외 (Networks 2026)을 확인하지 못했다. 유사 구조일 가능성이 있다.

**G3. 분해 오차의 정량화.** 노드 삭제 차분(= discrete semigradient [27])의 변환 오차를
**시뮬레이션 기준치와 그 표준오차 대비로** 재고, 편향이 **공급자 쪽에서 체계적으로 과소평가 방향**임을
묶음 보완성(초모듈성)과 연결하는 것. 한계 자체는 이미 지적되어 있으나([28][29][27]),
NeurADP [32]는 같은 분해를 하면서 오차를 측정하지 않는다.
**주의**: "처음 발견한 현상"으로 쓸 수 없다. "기존에 지적된 한계의 정량화"로 써야 한다.

**G4. 반응함수 오지정 강건성.** A3에서 확인한 공백이다. 보류해 둔 실험 2가 여기 놓인다.

**G5. OCL의 가드 장치를 ADP 반복 루프로.** [37][22]의 신뢰영역을 매 기간 재해결 구조에 가져오는 것.
우리 현재 가드는 사후 절단이므로 격차가 있다. 우선순위는 낮다.

### 죽은 갭 (주장하면 안 된다)

- "학습된 V를 MILP가 직접 최대화한다" → [18][20][19][21] 4편 선례.
- "비분리형 VFA를 IP에 넣는다" → 같은 4편. 방식은 모두 ReLU big-M 정확 임베딩.
- "분리형 근사가 상호작용에서 부정확하다" → [28][29][27]이 이미 지적.
- "분리형 vs 비분리형 VFA 비교" → [20]이 Transportation Science에서 이미 했다.

### 반박 대응표

| 반박 | 근거가 될 논문 | 우리 방어 |
|---|---|---|
| 잉여배분 동적 결정은 이미 있다 | Balseiro 외 [11], Chen 외 [12] | [11]은 참여자 집합 불변 + 하드 IR, [12]는 단일 스칼라 + 집계 도착률 |
| 이탈 내생 ADP는 이미 있다 | Luy 외 [3] | 결정변수가 채용 수, VFA가 분리형 PL, 반응이 금전 아님 |
| 학습 VFA의 MILP 임베딩은 이미 있다 | [18][20][19] | 인정한다. 우리 기여는 G2의 결정 의존 풀링과 G1의 설정이다 |
| 비분리성이 있어도 분리형이 잘 된다 | Topaloglu–Powell [33] | [29]의 다중 자원 종류 진술 + 우리 변환 오차 수치. **실험 1 결과는 [33] 쪽이므로 정직하게 그렇게 보고한다** |
| GNN이 필요 없다(그래프 고정 → 밀집망 동등) | Zhang 외 [35] | [35]은 *최적화 정식화*에 관한 것이고 *예측기로서의 귀납적 편향*과 별개라고 우리는 해석한다(전문 미확인, 초록·인용 문장 기준). 이 해석에 따라 헤드만 임베딩한다 |
| 5PL·미들마일 연구도 있다 | [39][40] | 개념·역량 논의이며 최적화 모형이 아니다 |

---

## 6. 확인하지 못한 것 (도서관 접속 우선순위)

| 순위 | 항목 | 왜 중요한가 |
|---|---|---|
| 1 | Hildebrandt, Bode, Ulmer, Mattfeld (*Networks* 2026) [21] — Wiley 403 | B2의 최근접 선례. MILP 정식화가 big-M인가 이상적인가, 망 크기, 해 시간, 최적화가 가치함수 오차를 착취하는 현상 보고 여부. **G2의 신규성을 직접 위협** |
| 2 | Delarue, Lian, Qin (2026 SSRN) [41] — SSRN 403 | 보수 결정이 기사 유지를 통해 미래에 영향을 주는 동적 모형인가. **그렇다면 A2·G1의 답이 바뀐다** |
| 3 | NeurADP [32] 전문 | 분해 오차를 어디서든 정량화했는가. **G3의 선행성 판정에 직결** |
| 4 | Luy 외 (2024) POM [3] | 완전정보 lookahead 벤치마크의 정의와 가려진 수치, PL-VFA의 분리 가능성 가정이 필요한 위치 |
| 5 | Chen, Zheng, Ke, Yang (2020) TR-B [12] — ScienceDirect 403 | 수수료율이 상태의존 정책인가, VFA 형태, 도착률 반응 함수형, 비교군 |
| 6 | Topaloglu–Powell (*IJOC* 2006) [33] | 분리형 근사가 모든 대체 패턴에서 잘 작동한 이유에 대한 저자 설명. 우리 반박 대응의 핵심 논거 |
| 7 | Musalem 외 (2023) *Operations Research* [15], Masorgo 외 (2026) *JOM* [17] | 반응함수 추정의 실증 앵커와 "금전만이 아니다" 반론의 강도 |
| 8 | Shi 외 (*IJOC* 2024) [37], Anderson 외 (*MP* 2020) [26], Maragno 외 (OR 2025) [22] | 신뢰영역 정식화 비용, 이상적 정식화의 해 시간 배수 |
| 9 | 협력 운송 이익배분(Shapley) 문헌 | 다기간에 배분이 다음 기간 연합 구성을 바꾸는 모형이 있으면 G1이 약해진다 |

기타 미해결: van Heeswijk·La Poutré [18]의 게재 여부, VDN [30]의 가산성 가정 명시 위치,
폴리헤드럴 서베이 [42]의 게재지 확정, A5 "유체 극한" 항목의 근거 보강, 저널 사분위 전수 확인.

---

## 7. 시뮬레이터·실험 설계에 대한 시사점

1. **후견 상한은 필수다.** 빈 `src/match/hindsight.py`를 채우고 모든 결과 표에 상한 대비 격차를 싣는다.
   근거: [9][8][3]. 근시안이 규칙 기반과 가치 근사를 모두 이기는 현 이상 현상도 상한 없이는 해석할 수 없다.
   Luy 외처럼 **완전정보 lookahead**를 비교군에 추가하는 것이 문헌 표준에 가장 가깝다.
2. **기여 서술을 재배치한다.** "학습 VFA를 MILP에 결합"을 방법론 기여로 내세우면 [18][20][19][21]에 걸린다.
   기여는 G1(설정) + G2(결정 의존 풀링) + G3(분해 오차 정량화)로 옮긴다. 초록 문언은 그대로 유지된다 —
   "GNN 기반 가치함수 근사를 MILP 기반 매칭·잉여배분 모형과 결합"이 G2의 서술이기 때문이다.
3. **실험 1의 음의 결과를 [33] 쪽 증거로 정직하게 위치시킨다.** 선형 ≈ MLP는 Topaloglu–Powell과 일치하고
   Powell–Topaloglu 튜토리얼의 다중 자원 종류 진술과 긴장한다. 그 긴장이 해소되는 조건을 재는 것이 실험 3이다.
4. **실험 3에서 [35]을 선제 대응한다.** 그래프가 결정 무관이므로 GNN 전용 정식화는 쓰지 않고 헤드만 임베딩한다는
   설계 근거를 명시한다. GNN 전체 임베딩은 [34]의 타임아웃 근거로 배제한다.
5. **실험 2의 우선순위를 올린다.** A3에서 확인한 공백(반응함수 오지정 강건성 연구 없음)이 G4다.
   보류 상태를 유지할 이유가 약해졌다.
6. **가드를 점검한다.** `coefs()`의 사후 절단이 공급자 c의 체계적 과소평가와 어떻게 상호작용하는지 재고,
   [37][22]의 신뢰영역과의 격차를 한계로 명시한다.
7. **계산 계획**: 헤드만 임베딩(ReLU 이진 16–32개)은 보고 범위 안이다. 단 결합 규모 해 시간의 선례가 없으므로
   구현 시 해 시간을 직접 측정해 보고해야 한다. z의 해석적 상·하한으로 big-M을 조인다.

---

## 8. 통합 Sources (검증판, [1]–[42])

번호는 이 문서 본문에서 인용된 순서로 새로 매겼다. 원자료에는 있으나 본문에서 실제로 인용되지 않은 항목
(예: 원 B[2], B[8], B[14], B[22], B[27], B[28], 원 A[4][5][8][15][18][19][20][24][27][28][29][30][34] 등)은
고아 출처를 만들지 않기 위해 이 목록에서 제외했다. 대응표는 `## 부록: 원 번호 대응표`에 있다.

읽은 수준: `전문`=본문 해당 절을 직접 추출해 읽음, `초록`=초록 전문을 직접 확보, `서지`=서지정보만 확인(내용 추정 금지).
URL 상태는 이 검증판 작성 시(2026-09-28) WebFetch로 재확인한 결과다.

1. Gans, N., Zhou, Y.-P. (2002) *Managing Learning and Turnover in Employee Staffing*, Operations Research — https://doi.org/10.1287/opre.50.6.991.343 — 서지 — 접근 불가(403, INFORMS 봇 차단), 서지 정보는 원자료(연구 A) 기준
2. Arlotto, A., Chick, S. E., Gans, N. (2014) *Optimal Hiring and Retention Policies for Heterogeneous Workers Who Learn*, Management Science — https://doi.org/10.1287/mnsc.2013.1754 — 서지 — 접근 불가(403, INFORMS 봇 차단), 확인일 2026-09-28
3. Luy, J., Hiermann, G., Schiffer, M. (2024) *Strategic Workforce Planning in Crowdsourced Delivery With Hybrid Driver Fleets*, Production and Operations Management — https://doi.org/10.1177/10591478241268602 · arXiv https://arxiv.org/abs/2311.17935 — 전문(arXiv판 모델·실험 절) — arXiv 링크 라이브 확인(제목·저자·제출일 일치, 2026-09-28). SAGE 게재판 페이지는 별도로 확인하지 않았다
4. Taylor, T. A. (2018) *On-Demand Service Platforms*, M&SOM — https://doi.org/10.1287/msom.2017.0678 — 초록 — 접근 불가(403, INFORMS 봇 차단)
5. Cachon, G. P., Daniels, K. M., Lobel, R. (2017) *The Role of Surge Pricing on a Service Platform with Self-Scheduling Capacity*, M&SOM — https://doi.org/10.1287/msom.2017.0618 — 초록 — 접근 불가(403, INFORMS 봇 차단)
6. Bhargava, H. K., Wang, K., Zhang, X. (2022) *Fending Off Critics of Platform Power with Differential Revenue Sharing*, Management Science — https://doi.org/10.1287/mnsc.2022.4545 — 초록 — 접근 불가(403, INFORMS 봇 차단)
7. Hu, M., Zhou, Y. (2022) *Dynamic Type Matching*, M&SOM — https://doi.org/10.1287/msom.2020.0952 (arXiv https://arxiv.org/abs/1811.07048) — 초록 — 정식 게재판 접근 불가(403). arXiv 대체본은 이번 검증에서 별도로 열어보지 않았음(원자료 기준 존재)
8. Aouad, A., Sarıtaç, Ö. (2022) *Dynamic Stochastic Matching Under Limited Time*, Operations Research — https://doi.org/10.1287/opre.2022.2293 — 초록 — 접근 불가(403, INFORMS 봇 차단)
9. You, F., Vossen, T. W. M. (2024) *An Approximate Dynamic Programming Approach to Dynamic Stochastic Matching*, INFORMS Journal on Computing — https://doi.org/10.1287/ijoc.2021.0203 — 초록 — 접근 불가(403, INFORMS 봇 차단)
10. Hellemo, L., Barton, P. I., Tomasgård, A. (2018) *Decision-dependent probabilities in stochastic programs with recourse*, Computational Management Science — https://doi.org/10.1007/s10287-018-0330-0 — 서지 — 접근 불가(403/로그인 요구, Springer)
11. Balseiro, S., Lin, M. S., Mirrokni, V., Paes Leme, R., Zuo, S. (2017) *Dynamic Revenue Sharing*, NeurIPS 2017 — https://papers.nips.cc/paper/2017/file/cb8acb1dc9821bf74e6ca9068032d623-Paper.pdf — 전문 — 라이브 확인(PDF 열람, 제목·저자·게재처 일치, 2026-09-28)
12. Chen, X., Zheng, H., Ke, J., Yang, H. (2020) *Dynamic optimization strategies for on-demand ride services platform: Surge pricing, commission rate, and incentives*, Transportation Research Part B 138:23–45 — https://doi.org/10.1016/j.trb.2020.05.005 — 초록(검색 요약 경유, medium confidence) — 접근 불가(Elsevier가 자동 리디렉션 안내 페이지만 반환, 실제 콘텐츠 미확인 — 봇 차단으로 추정)
13. Schur, R., Winheller, K. (2024) *Optimizing last-mile delivery: a dynamic compensation strategy for occasional drivers*, OR Spectrum — https://doi.org/10.1007/s00291-024-00796-6 — 초록 — 접근 불가(403/로그인 요구, Springer)
14. Chen, L., Xu, B. (2026, 프리프린트, 미심사) *Fairness as an Investment: Dynamic Participation and Long-Run Profit in Virtual Power Plants*, arXiv — https://arxiv.org/abs/2606.02820 — 전문(모델·벤치마크 절) — 라이브 확인(제목·저자·제출일 일치, 2026-09-28)
15. Musalem, A., Olivares, M., Yung, D. (2023) *Balancing Agent Retention and Waiting Time in Service Platforms*, Operations Research 71(3):979–1003 — https://doi.org/10.1287/opre.2022.2418 — 초록 — 접근 불가(403, INFORMS 봇 차단, 리디렉션 확인)
16. Xu, S., Zhang, Y., Miller, E. J. (2025, 프리프린트) *Frailty-Aware Transformer for Recurrent Survival Modeling of Driver Retention in Ride-Hailing Platforms*, arXiv — https://arxiv.org/abs/2511.19893 — 서지 — 라이브 확인(제목·저자 일치, KDD Workshop 2025 채택 명시, 2026-09-28)
17. Masorgo, N., Dobrzykowski, D. D., Tang, C. S., Fugate, B. S. (2026) *There Is More to Crowdshipping Than Money*, Journal of Operations Management — https://doi.org/10.1002/joom.70046 — 서지 — 접근 불가(403, Wiley 봇 차단)
18. van Heeswijk, W., La Poutré, H. (2019) *Approximate Dynamic Programming with Neural Networks in Linear Discrete Action Spaces*, arXiv (게재 학회/저널 미확인) — https://arxiv.org/abs/1902.09855 · ar5iv https://ar5iv.labs.arxiv.org/abs/1902.09855 — 전문(ar5iv) — 라이브 확인(양쪽 다, 2026-09-28)
19. Delarue, A., Anderson, R., Tjandraatmadja, C. (2020) *Reinforcement Learning with Combinatorial Actions: An Application to Vehicle Routing*, NeurIPS 2020 — https://proceedings.neurips.cc/paper/2020/file/06a9d51e04213572ef0720dd27a84792-Paper.pdf · ar5iv https://ar5iv.labs.arxiv.org/html/2010.12001 — 전문(ar5iv) — 공식 PDF는 라이브(다운로드됨, 텍스트 자동추출은 실패)이고 ar5iv 대체본에서 제목을 직접 확인(2026-09-28)
20. van Steenbergen, R., van Heeswijk, W., Mes, M. (2025) *The Stochastic Dynamic Postdisaster Inventory Allocation Problem with Trucks and UAVs*, Transportation Science 59(2):360–390 — https://doi.org/10.1287/trsc.2023.0438 · arXiv https://arxiv.org/abs/2312.00140 — 전문(HTML 해당 절) — 게재판 DOI는 접근 불가(403, INFORMS), arXiv 대체본은 라이브 확인(제목·저자·결과 요약 일치, 2026-09-28)
21. Hildebrandt, F., Bode, A., Ulmer, M., Mattfeld, D. (2026) *Shaping Decision Models for Stochastic Dynamic Optimization Problems via Reinforcement Learning*, Networks — https://doi.org/10.1002/net.70039 — 초록만 — 접근 불가(403, Wiley 봇 차단). B2(§3.1)의 최근접 선례이므로 확보 시 결론이 바뀔 수 있음(§6 우선순위 1)
22. Maragno, D., Wiberg, H., Bertsimas, D., Birbil, Ş. İ., den Hertog, D., Fajemisin, A. (2025) *Mixed-Integer Optimization with Constraint Learning*, Operations Research 73(2):1011–1028 — https://doi.org/10.1287/opre.2021.0707 · arXiv https://arxiv.org/abs/2111.04469 — 초록·서지 + 공식 도구 설명 — 게재판 접근 불가(403, INFORMS), arXiv 프리프린트는 라이브 확인(2026-09-28)
23. Bergman, D., Huang, T., Brooks, P., Lodi, A., Raghunathan, A. (2022) *JANOS: An Integrated Predictive and Prescriptive Modeling Framework*, INFORMS Journal on Computing — https://doi.org/10.1287/ijoc.2020.1023 — 서지 — 접근 불가(403, INFORMS 봇 차단)
24. Ceccon, F., Jalving, J., Haddad, J., Thebelt, A., Tsay, C., Laird, C. D., Misener, R. (2022) *OMLT: Optimization & Machine Learning Toolkit*, JMLR 23 — https://www.jmlr.org/papers/volume23/22-0277/22-0277.pdf — 서지·스니펫 — PDF 라이브(오픈 액세스 호스트, 다운로드됨). 텍스트 자동추출은 제목을 재현하지 못했으나 JMLR은 무료 공개 저널이라 접근 자체는 문제없음
25. Gurobi *Machine Learning — Supported models* (공식 문서, 접속 2026-09-28) — https://gurobi-machinelearning.readthedocs.io/en/stable/user/supported.html — 전문(해당 페이지) — 라이브 확인(지원 모델 목록 내용 일치)
26. Anderson, R., Huchette, J., Ma, W., Tjandraatmadja, C., Vielma, J. P. (2020) *Strong mixed-integer programming formulations for trained neural networks*, Mathematical Programming — https://doi.org/10.1007/s10107-020-01474-5 — 서지 — 접근 불가(403/로그인 요구, Springer)
27. Iyer, R., Jegelka, S., Bilmes, J. (2013) *Fast Semidifferential-based Submodular Function Optimization*, ICML 2013 — http://proceedings.mlr.press/v28/iyer13.pdf — 전문 부분 추출 — PDF 라이브(오픈 액세스 MLR Proceedings 호스트, 다운로드됨). 텍스트 자동추출은 제목을 재현하지 못했으나 호스트 자체는 문제없음
28. Powell, W. B., Ruszczyński, A., Topaloglu, H. (2004) *Learning Algorithms for Separable Approximations of Discrete Stochastic Optimization Problems*, Mathematics of Operations Research 29(4):814–836 — https://doi.org/10.1287/moor.1040.0107 · PDF https://people.orie.cornell.edu/huseyin/publications/spar.pdf — 전문 부분 추출 — 게재판은 INFORMS 패턴상 접근 불가로 추정(개별 재확인 안 함). 저자 개인 페이지 PDF는 라이브(다운로드됨, 자동추출로 제목 재현은 못했으나 호스트가 저자 본인 사이트라 신뢰)
29. Powell, W. B., Topaloglu, H. *Approximate Dynamic Programming for Large-Scale Resource Allocation Problems* (INFORMS TutORials, 2005) — https://people.orie.cornell.edu/huseyin/publications/tutorial_powell_topaloglu_2.pdf — 전문 부분 추출 — 라이브 확인(제목·저자 재현됨, 2026-09-28)
30. Sunehag, P. 외 10인 (2017) *Value-Decomposition Networks For Cooperative Multi-Agent Learning*, arXiv — https://arxiv.org/abs/1706.05296 — 초록 — 라이브 확인(제목·전체 저자 목록 재현, 2026-09-28). 학회 게재 여부는 이번 검증에서도 확인하지 못함
31. Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., Whiteson, S. (2018) *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018 — https://arxiv.org/abs/1705.08926 — 서지·이차 요약 — 라이브 확인(arXiv판, 2026-09-28)
32. Shah, S., Lowalekar, M., Varakantham, P. (2020) *Neural Approximate Dynamic Programming for On-Demand Ride-Pooling*, AAAI 2020 — https://arxiv.org/abs/1911.08842 — 전문의 분해 절 — 라이브 확인(2026-09-28)
33. Topaloglu, H., Powell, W. B. (2006) *Dynamic-Programming Approximations for Stochastic, Time-Staged Integer Multicommodity-Flow Problems*, INFORMS Journal on Computing 18(1):31–42 — https://doi.org/10.1287/ijoc.1040.0079 · PDF https://people.orie.cornell.edu/huseyin/publications/imcf.pdf — 전문 부분 추출 — 게재판 접근 불가(추정), 저자 페이지 PDF 라이브(제목·저자 재현됨, 2026-09-28)
34. Elrefaei, J. D., Hua, K., Kim, S., Tran, H. N., Borrero, J. S. (2026, 프리프린트, 미게재) *Graph4BiLO: Graph Neural Network Approximation for Bilevel Mixed-Integer Linear Optimization*, arXiv — https://arxiv.org/abs/2608.30103 — 전문 HTML — 라이브 확인(2026-09-28)
35. Zhang, S., Campos, J. S., Feldmann, C., Walz, D., Sandfort, F., Mathea, M., Tsay, C., Misener, R. (2023) *Optimizing over trained GNNs via symmetry breaking*, NeurIPS 2023 — https://arxiv.org/abs/2305.09420 — 초록 + 인용 문장만(전문 미확인) — 라이브 확인(2026-09-28). §3.3·§5에서 이 논문의 사정거리를 논한 부분은 우리 해석이며 전문 대조 전까지는 잠정적이다
36. Okada, G., Noda, S., Komiyama, J., Matsushita, A. (2026, 프리프린트, 미게재) *Learning Optimal Dynamic Matching via Graph Neural Networks*, arXiv — https://arxiv.org/abs/2607.28925 — 초록 — 라이브 확인(2026-09-28)
37. Shi, C., Emadikhiav, M., Lozano, L., Bergman, D. (2024) *Constraint Learning to Define Trust Regions in Optimization over Pre-Trained Predictive Models*, INFORMS Journal on Computing 36(6):1382–1399 — https://doi.org/10.1287/ijoc.2022.0312 · 프리프린트 https://arxiv.org/abs/2201.04429 — 초록 — 게재판 접근 불가(403, INFORMS), arXiv 프리프린트는 라이브 확인(2026-09-28)
38. Smith, J. E., Winkler, R. L. (2006) *The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis*, Management Science 52(3):311–322 — https://doi.org/10.1287/mnsc.1050.0451 — 초록·이차 요약 — 접근 불가(403, INFORMS 봇 차단, 리디렉션 확인)
39. Hosie, P. J., Sundarakani, B., Tan, A. W. K. (2012) *Determinants of fifth party logistics (5PL): service providers for supply chain management*, International Journal of Logistics Systems and Management — https://doi.org/10.1504/ijlsm.2012.049700 — 서지 — 접근 불가(403, Inderscience Online 봇 차단, 리디렉션 확인)
40. Nicoletti, B., Appolloni, A. (2024) *Digital transformation in ecosystems: integrated operations model and its application to fifth-party logistics operators*, Journal of Global Operations and Strategic Sourcing 18(1) — https://doi.org/10.1108/jgoss-04-2023-0024 — 서지 — 라이브 확인(Emerald 페이지에서 초록·목차 전체 열람 가능, 2026-09-28) — 원자료보다 접근성이 좋아진 사례
41. Delarue, A., Lian, Z., Qin, Z. T. (2026, 워킹페이퍼) *Learning Pay Strategies with Small Samples in Gig Economy Platforms*, SSRN — https://doi.org/10.2139/ssrn.6197638 — 서지 — 접근 불가(403, SSRN 봇 차단). G1의 위험 요인(§5)이자 여전히 미해결
42. Huchette, J., Muñoz, G., Serra, T., Tsay, C. *When Deep Learning Meets Polyhedral Theory: A Survey*, arXiv (게재지 표기 상충 — 미확인) — https://arxiv.org/abs/2305.00241 — 전문 부분 grep — 라이브 확인(2026-09-28)

---

## 부록: 원 번호 대응표

이 표는 검증 과정 기록용이며 본문 인용에는 쓰지 않는다. "원 번호"는 `A[n]`/`B[n]` 표기다.

| 새 번호 | 원 번호 | 새 번호 | 원 번호 | 새 번호 | 원 번호 |
|---|---|---|---|---|---|
| [1] A[22] | [15] A[9] | [29] B[19] |
| [2] A[21] | [16] A[25] | [30] B[25] |
| [3] A[10] | [17] A[26] | [31] B[26] |
| [4] A[1]  | [18] B[1]  | [32] B[5] |
| [5] A[2]  | [19] B[6]  | [33] B[18] |
| [6] A[14] | [20] B[3]  | [34] B[23] |
| [7] A[6]  | [21] B[4]  | [35] B[21] |
| [8] A[7]  | [22] B[7]  | [36] B[24] |
| [9] A[11] | [23] B[10] | [37] B[15] |
| [10] A[23]| [24] B[11] | [38] B[16] |
| [11] A[13]| [25] B[12] | [39] A[32] |
| [12] A[12]| [26] B[9]  | [40] A[33] |
| [13] A[16]| [27] B[20] | [41] A[31] |
| [14] A[17]| [28] B[17] | [42] B[13] |

원자료에는 있으나 이 검증판 본문에서 인용되지 않아 Sources에서 제외한 항목: A[3][4][5][8][15][18][19][20][24][27][28][29][30][34],
B[2][8][14][22][27][28]. 이 중 확인 과정에서 새로 밝혀진 사실은 다음과 같다(참고용, 본문 미반영):
- A[34](저자 미확인으로 기록됐던 job-matching 프리프린트)의 저자는 Tatsuya Ute, Chiaki Ichimura, Yuta Saito로 확인됐다(arXiv 2609.01652).
- B[2](van Heeswijk & La Poutré 2020, "게재 미확인"으로 기록됨)는 OpenAlex 메타데이터상 2020 Winter Simulation Conference(WSC)
  발표 논문(DOI 10.1109/wsc48552.2020.9384078)으로 확인됐다. IEEE Xplore 페이지 자체는 이번 검증에서 열람하지 못했다.
- B[27](Conformal Mixed-Integer Constraint Learning)의 저자는 Ovalle, Biegler, Grossmann, Laird, Dulce Rubio로 확인됐다.
- B[28](PySCIPOpt-ML)의 저자는 Turner, Chmiela, Koch, Winkler로 확인됐다.
- A[30](Bimpikis, Candogan, Sabán)에 대한 원자료의 경고("OpenAlex DOI가 ACM EC 2017판과 뒤섞여 있다")는 OpenAlex API 원본
  JSON(`https://api.openalex.org/works/W2766210608`)으로 재확인했다: `doi`는 `10.1145/3106723.3106728`(ACM EC)이지만
  `primary_location.source.display_name`은 "Operations Research"로 표기되어 있어, 원자료의 지적이 **정확했다.**
