# 내생적 참여자 구성과 학습 가치함수의 분리형 근사: 문헌 검토와 갭 특정

작성일: 2026-09-28 · slug `endogenous-participation-vfa-embedding`
범위 문서: `ref/literature-review.md` (1차 A·B / 2차 C·D·E / 3차 F)
원자료: `ref/_review/..._research_{A,B,C,D,E,F1,F2,F3,F4}.md` (증거 약 200건)
감사: `ref/_review/..._review.md` (적대적 감사, 치명적 5건) · 인용 검증: `..._cited.md`

이 문서는 문헌 요약이 아니라 **갭 특정**이다. 결론부터 적는다.

**우리가 당초 기여로 지목한 것 중 세 개가 선례로 채워져 있다.** (1) 학습된 가치함수를 MILP가 직접
최대화하는 것, (2) 결정변수에 의존하는 원소별 임베딩을 풀링해 헤드만 임베딩하는 것, (3) 참여자별
증분 가치를 최적화 계수로 쓰는 것. 그리고 우리가 "부호 반전"으로 내세우려던 현상(증분 가치 > 독립 가치)도
**서로 다른 세 문헌에 이미 있다.**

**남는 것은 그 현상이 얹히는 대상과, 우리가 실제로 측정할 수 있는 것이다.** 기존 문헌의 분리형 근사는
모두 **외생 용량·재고 격자** 위에 있고 계수 가중치가 **상태 의존일 뿐 결정 의존이 아니다.** 우리는
**결정에 따라 전이하는 참여자 집합** 위에 있고 가중치가 **결정변수(재참여확률)** 다. 이것이 유일하게
방어 가능한 자리이며, 그 위에서 분리형 근사의 오차를 **부호가 사전 예측된 형태로 측정**할 수 있다.

---

## 1. 조사 방법과 한계

WebSearch로 지형을 잡고 **인용 그래프 순회**(OpenAlex, 보조로 Crossref·OpenCitations·Semantic Scholar)를
주 수단으로 썼다. 키워드 검색만으로는 이 주제가 잡히지 않는다 — 착수 시 시험 쿼리의 1위 결과가 무관한
서베이(*Knowledge Graphs*, 인용 1835)였다. 씨앗 논문의 참조·피인용을 교차해 **공통 참조**와
**두 군집의 공통 인용자가 0인 지점**을 갭 신호로 삼았다.

한계를 먼저 밝힌다. 이것들은 아래 판정의 신뢰도를 직접 제약한다.

- **저널 사분위를 직접 확인한 항목이 없다.** 원자료에서 전부 "사분위 미확인"으로 표기했다.
- **전문을 읽은 논문은 소수다.** 주제별로 3~8편이고, 나머지는 초록 또는 서지 수준이다. 원자료는 항목마다
  읽은 수준(전문/초록/서지만/검색 수준)을 표기했다. 이 문서에서 판정 근거로 쓴 인용은 대부분 전문 또는
  초록 원문에서 추출한 것이고, 그렇지 않은 경우 본문에 표시했다.
- **주요 출판사가 전부 봇 차단이다.** INFORMS(pubsonline)·Elsevier(ScienceDirect)·Wiley·SSRN·
  Inderscience가 일관되게 HTTP 403이다. 우회 경로(저자 사본 PDF, arXiv, RePEc, 기관 리포지터리,
  Crossref 등록 초록, OpenAlex `abstract_inverted_index` 복원)로 상당수를 메웠으나 메우지 못한 것이 남았다(§6).
- **OpenAlex 인용 그래프 전수 순회를 완주하지 못했다.** 익명 호출에 두 종류의 429가 있었다:
  검색 클러스터 일시 스로틀(retryAfter 31초)과 **일일 예산 소진**(`"Insufficient budget ... $0 remaining;
  resets at midnight UTC"`, retryAfter 약 14.6시간). 단일 레코드 경로(`works/doi:`, `works/W<id>`)는
  예산 소진 중에도 통과했으므로 참조 목록 역순회는 되었고, **전방 피인용 순회가 부분적으로 실패**했다.
  OpenCitations로 대체한 구간이 있다(커버리지 약 20% 누락 추정). UTC 자정 이후 재시도 과제로 남겼다.
- **"0건"은 "존재하지 않음"이 아니다.** 이 색인과 이 쿼리로 걸리지 않았다는 뜻이다. 원자료에 0건 쿼리를
  전부 남겼다.
- **도구 신뢰성 사고 1건**: WebFetch가 저자 출판목록 페이지에서 `example.com` 형태의 조작 URL을 반환했다.
  해당 조사자가 전부 폐기하고 실제 URL을 HTTP 200으로 재검증했다. 이 문서의 URL은 검증된 것만 쓴다.
- **절차 위반 1건(기록)**: 조사 도중 두 에이전트가 OpenAlex·Crossref·Unpaywall 호출에 사용자 이메일을
  `mailto`/`email` 파라미터로 넣었다(polite pool 관행이나 **사전 허락 없이** 한 것). 금지 지시 이후
  중단했고, 이후 호출은 이메일 없이 수행해 정상 응답을 받았다. 산출물 URL에 이메일은 남지 않았다.

---

## 2. 주제별 종합

### 2.1 A — 내생적 참여자 구성: 세 계보가 서로의 방법을 쓰지 않는다

| 계보 | 대표 | 참여 내생 | 기간 간 전이 | 금전 배분 결정 |
|---|---|---|---|---|
| 인력계획 MDP | Gans·Zhou 2002 → Arlotto 외 2014 → **Luy·Hiermann·Schiffer 2024 (POM)** | O | O | X |
| 플랫폼 경제학 | Taylor 2018, Cachon 외 2017, Bhargava 외 2022 | O | **X (정태 균형)** | O |
| 동적 매칭 | Hu·Zhou 2022, Aouad·Sarıtaç 2022 (OR), You·Vossen 2024 | 부분(**외생 이탈**) | O | X |

방법론 라벨은 **decision-dependent (endogenous) uncertainty**다.

**A2 직답 — 세 갈래로 나뉘어 있다.** (1) 잉여배분 몫을 기간별 결정변수로 둔 동적 모형은 있다
(Balseiro 외 NeurIPS 2017) — 그러나 참여가 기대값 IR 제약이고 **참여자 집합이 변하지 않는다.**
(2) 수수료율을 기간별로 결정하고 ADP로 푼 모형도 있다(Chen 외 2020 TR-B) — 그러나 **시장 전체 단일
스칼라**이고 반응이 **집계 도착률**이어서 "참여자 i를 잃는 손해"라는 개념이 성립하지 않는다.
(3) 참여자 이탈이 결정에 내생인 ADP도 있다(Luy 외 2024 POM) — 그러나 결정변수가 **채용 수**이고
이탈이 금전이 아니라 **미매칭 비율**에 반응한다.

**Luy 외 (2024) POM은 반드시 인용·차별화해야 한다.** arXiv 본문 3.3.2절에서 확인: VFA가
**"고정기사 차원의 분리형 구간선형(PL-VFA), 오목성 유지"** 다 — 우리가 "기존 ADP 표준 근사구조"로
지목한 바로 그 형태가 2024년 POM 논문의 실제 선택이다. 이탈확률은 미매칭 비율의 두 점 선형 혼합이고,
비교군에 **완전정보 lookahead**가 있다.

**A5 — 우리 비교군에 상한이 없는 것이 문헌 기준 미달이다.** 표준은 LP/ALP 완화 상한과 최적성 격차
(You·Vossen 2024, Aouad·Sarıtaç 2022는 LP 벤치마크 + 상수비 보증) 또는 완전정보 상한(Luy 외)이다.
우리 비교군은 전부 정책이다. 단 §7에 적은 유보가 있다.

**A4 — 응용 맥락은 공백이다.** 5PL 문헌(OpenAlex 72건)은 개념·역량 논의이고 최적화 모형이 아니다.
`middle mile consolidation hub dynamic optimization platform` 0건,
`all-or-nothing bundle order platform dynamic allocation supplier` 0건,
`B2B platform dynamic matching suppliers buyers` 0건.

### 2.2 B·C — 학습 가치함수의 MILP 임베딩: 기여가 아니다

**학습된 비분리형 신경망 VFA를 매 기간 MILP에 big-M으로 정확 임베딩하고 ADP로 반복 해결한 연구가 있다.**

| 연구 | 게재 | 규모·해 시간 | 읽은 수준 |
|---|---|---|---|
| Delarue, Anderson, Tjandraatmadja 2020 | NeurIPS | 은닉 1층 16뉴런, Gurobi 0.4초(입력 21) / 39초(입력 51) | 전문 |
| van Steenbergen, van Heeswijk, Mes 2025 | **Transportation Science** 59(2) | 0.03–0.20초/에폭, **분리형 VFA와 직접 비교** | 전문(해당 절) |
| van Heeswijk·La Poutré | **WSC 2020** (DOI 10.1109/wsc48552.2020.9384078) | 은닉 3×20에서 반복당 0.39초 | 전문(ar5iv) |
| Hildebrandt, Bode, Ulmer, Mattfeld 2026 | **Networks** (오픈액세스 CC-BY) | 미확인 | 초록만 |

특히 van Steenbergen 외는 **비분리형 NN-VFA를 분리형 VFA와 직접 비교한 Transportation Science 논문**이므로,
"분리형 vs 비분리형 비교"라는 프레이밍도 단독으로는 기여가 되지 않는다.

**결정 의존 풀링(우리가 대안 설계로 검토한 구조)에도 직접 선례가 있다.** Neur2RO(ICLR 2024,
arXiv:2310.04345) 전문에서 직접 확인:

> "embeddings are computed for each single first-stage and scenario variable (xᵢ and ξᵢ) ... These
> embeddings are then aggregated and passed through an additional feed-forward neural network."
>
> "we only have to represent the embedding network Φx and the small value network Φ ... Since the scenario
> parameters are not variables here, the scenario embeddings can be precomputed via a forward pass ...
> **no MILP representation is needed for Φξ**."

gurobi-machinelearning 1.3.0으로 임베딩한다. 즉 "결정 의존 부분만 임베딩하고 결정 무관 부분은 사전 계산"
원칙까지 같은 논문에 있다. **Neur2BiLO(NeurIPS 2024)는 더 나아가 합 풀링 항이 연속 결정변수에 의존하는
사례를 포함한다.** **SEQUOIA/coRMAB(ICLR 2025)** 는 학습 Q망을 매 기간 MILP에 임베딩해 이분 매칭·용량
제약이 있는 조합 행동을 고르는 다기간 루프를 이미 한다.

우리 구조(`z = Σ p_i h_i`, h_i 상수)는 이 계보의 **선형 특수형이자 더 쉬운 경우**다. 방법론 기여로 내세울 수 없다.

**C7 — 이 계보가 비워 둔 칸이 하나 있다.** Neur2SP·Neur2RO·Neur2BiLO·SEQUOIA **네 편 전문에
Jensen 격차와 학습/최적화 입력 분포 불일치를 다루는 서술이 없다.** 우리 설계에서 계산하려는 것은
`E[V(무작위 잔존 그래프)]`인데 계산하는 것은 `V(기대 그래프)`이고, 학습은 `p ∈ {0,1}` 실현 그래프로 하는데
최적화 시점에는 `p ∈ (0,1)` 소수값이 들어가 **학습 분포 밖**에서 평가된다. 형식 고리가 하나 있다:
**Wan 외, Planning with Expectation Models (IJCAI 2019)** — 기대 모형 계획이 분포 모형 계획과 동등한
조건은 **가치함수가 상태 특징에 선형일 때**다. 헤드가 비선형이면 격차가 남는다.

**C3 — 우리 GNN/MLP 비교는 현재 상태로 "공정 비교"라고 쓸 수 없다.** Errica 외(ICLR 2020) 표준
(외부 CV + 내부 holdout + 최종 재학습 + 넓은 그리드) 대비 빠진 것: (a) 외부 CV·시드 반복 없음
(분할 1개, 재학습 1회) → `conc1_k1`의 GNN R² 0.007이 분할 운인지 구분 불가, (b) 그리드가 weight decay
2개뿐이고 GNN 300초 vs MLP 1.4초라 같은 예산이 같은 탐색 폭이 아님, (c) **입력 전처리 동등화가 깨짐**
(공급자 수량 최대 |z| 10~11).

**C5 — 음의 결과 선례가 있다(우리 실험 3 v1의 방어 근거).** Errica 외(ICLR 2020)의 구조 무관 기준선은
**"노드 특징 합 풀링 + 1층 MLP"** 로 우리 시장 요약 지표 MLP와 사실상 같은 구조이고,
"on D&D, PROTEINS and ENZYMES none of the GNNs are able to improve over the baseline"라고 보고한다.
**Bechler-Speicher 외(ICML 2024)** 는 정답이 그래프를 쓰지 않을 때 GNN이 그래프를 쓰는 쪽으로 과적합하며
**무한 데이터에서도 보장이 없다**고 한다 — 우리 `items_per_order = 1` 대조 조건이 "차이 없음"이 아니라
**"GNN이 더 나쁨"** 방향으로 깨질 수 있다는 예측이고, 실제로 `conc1_k1`이 그 모양이다.

**Zhang 외(NeurIPS 2023)** 의 사정거리를 정확히 해 둘 필요가 있다: "고전적 GNN 구조에서 **그래프가
고정되면** GNN에 대한 최적화는 밀집 신경망에 대한 최적화와 동등하다." 이는 *최적화 정식화*에 관한 진술이고
*예측기로서의 귀납적 편향*과 별개다(이 구분은 원문 문장에 근거하지만 저자가 그렇게 프레이밍한 것은 아니다).
따라서 "GNN 전용 정식화를 만들지 말고 헤드만 임베딩하라"는 설계 근거로는 쓸 수 있으나,
**GNN을 쓰는 정당화는 전적으로 예측 성능에 달려 있다.** 그런데 우리 GNN은 네 칸 모두 검증 R² 최하위다
(`exp3_diag.json`: 0.376 / 0.279 / **0.007** / 0.279, 잡음 상한 0.426~0.487). 예측 성능을 회복하기 전에는
이 방어가 성립하지 않는다.

### 2.3 D·F3 — 분리형 근사의 오차: 부호는 이론적으로 미결이고, 측정 가능하다

**우리 `c_i`에는 확립된 이름이 셋 있다.**
- **discrete semigradient (supergradient)** — Iyer–Jegelka–Bilmes (ICML 2013). 이것이 정의하는 모듈러
  상·하한은 **현재 집합에서만 tight**하다(`m_ĝ(Y) = f(Y)`). 즉 참여자 일부가 떠난 집합에서 어긋나는 것은
  **정의상 예견된 바**이며 발견이 아니다.
- **다선형 확장의 1차 Taylor** — `F(p) = E[V(R(p))]`(Chekuri–Vondrák–Zenklusen). 우리 MILP 미래항
  `Σ c_i p_i`는 정확히 `F`의 `p = 1`에서의 1차 전개다.
- **Birnbaum 중요도를 전원 참여점에서 평가한 값** — `c_i = ∂F/∂p_i|_{p=1}` (F3).

**부호 방향이 문헌으로 확정된다. 그리고 우리가 쓰려던 방향과 반대다.**
- **열모듈(대체 지배)** → 가산 대리값이 축소 집합 가치를 과대평가 = **다수 이탈 손실을 과소평가**.
  `c_i`는 가능한 한계값 중 최소이므로 `c_i ≤ ∂F/∂p_i`.
- **초모듈(보완 지배)** → 반대로 **과대평가**.
- 신뢰도 이론이 같은 결론을 독립적으로 준다: 병렬(대체)에서 `I_B(i) = Π(1−p_j)`가 `p=1`에서 **최소**,
  직렬(보완)에서 `I_B(i) = Π p_j`가 `p=1`에서 **최대**.
- ML 데이터 귀속 쪽 대응: **Hu 외(NeurIPS 2024)** 는 개별 1차 영향의 합이 집합 효과를 못 잡는 실패가
  **선형회귀에서도 증명 가능**하다고 보고한다. Koh 외(NeurIPS 2019), Basu 외(ICML 2020)도 같은 방향이다.

**"우리 가치함수가 초모듈"이라는 주장은 쓸 수 없다.** 두 가지 이유가 겹친다.
1. **혼재**: `n_alt = 2`에서 같은 품목의 두 공급자는 용량 대체재(열모듈 방향), 다른 품목 공급자는
   보완재(초모듈 방향)다. Bian 외(ICML 2017)의 언어로 `γ<1` 이면서 `α>0`인 혼재 함수이고, 임의 집합함수의
   DS 분해가 항상 가능하므로 **부류 차원에서 순 부호가 결정되지 않는다.**
2. **정리 수준의 금지(이진 한정)**: Block–Griffith–Savits (1989) 초록이
   "L-superadditive functions are also known under the names **supermodular**"이며
   "For binary structure functions of binary values, El-Neweihi (1980) showed that L-superadditive structure
   functions **must be series**"라고 적는다. 우리 구조는 parallel-in-series이므로 순수 직렬이 아니고,
   따라서 **이진 근사에서는 초모듈일 수 없다.** 같은 초록이 "In the case of **non-binary-valued** structure
   functions this is no longer the case"라고 하므로, 연속 이윤인 우리 V에는 금지가 적용되지 않는다.
   정확한 서술은 **"이진 근사에서는 금지, 연속 가치함수에서는 열려 있음"** 이다.
   (두 논문 모두 초록만 읽었다. 인용 전 전문 확인이 필요하다.)

**단조성 전제는 저장된 원자료로 확인했다.** El-Neweihi·Block 외는 coherent system(단조)을 가정한다.
`exp1_value.json`·`exp3_mlp.json`의 반복별 `mc_buy`·`mc_sup`에서 표준오차 대비 한쪽 95% 기준으로:

| 칸 | 주문자 음수 | 주문자 유의 음수 | 공급자 음수 | 공급자 유의 음수 |
|---|---|---|---|---|
| exp1 | 58/345 (16.8%) | 3 (0.9%) | 0/45 (0.0%) | **0** |
| conc0_k1 | 43/381 (11.3%) | 1 (0.3%) | 2/48 (4.2%) | **0** |
| conc0_k2 | 46/357 (12.9%) | 0 (0.0%) | 1/42 (2.4%) | **0** |
| conc1_k1 | 49/360 (13.6%) | 3 (0.8%) | 2/43 (4.7%) | **0** |
| conc1_k2 | 58/349 (16.6%) | 1 (0.3%) | 1/43 (2.3%) | **0** |

**공급자 쪽은 다섯 칸 전부 유의 음수 0건**이고 주문자도 0~0.9%로 잡음 수준이다. 단조성 위반의 증거가 없다.
신뢰도 구조 논증이 적용되는 대상이 정확히 공급자이므로 전제가 충족된다.
(즉석 스크립트로 계산했다. 보고에 쓰려면 `result/diag.py`에 넣어 원자료에서 재생산되게 해야 한다.)

**주의 — 지금 저장소의 "변환 오차"는 이 사슬이 말하는 양이 아니다.** 모델 c와 기준치 c가 **둘 다
leave-one-out**이므로 모델 오차이지 분해 손실이 아니다. 관측된 공급자 c 과소평가는 위 부호 논증으로
설명되지 않으며, 경쟁 가설 넷(회귀 축소, **표본 불균형 42~48 대 349~381**, 가드 shrink·절단,
정답 잡음 R² 상한 0.43~0.49)이 더 그럴듯하다.

### 2.4 E — 참여자별 증분 가치를 최적화 계수로 쓰는 것: 선례가 있다

**E6 직답 — 있다. 단정적으로 그렇다.**
- **Klein & Kolb (2015) Omega 55:111–125**: 용량 수락/거절 MDP에서 "opportunity cost-based approach that
  understands customer profitability as a customer's **contribution to customer equity**". 같은 초록이
  "isolated determination and optimization of a single customer's lifetime value is no longer feasible"라고
  적어 상호의존 논거까지 선점한다. **다만 그것을 최적화 계수로 쓰는지 사후 지표로만 쓰는지는 판정 불가**
  (본문 접근 실패, §6 1순위).
- **Ovchinnikov, Boulu-Reshef, Pfeifer (2014) Management Science 60(8)**: **VIC(value of an incremental
  customer)** — 용량이 제한되면 VIC는 CLV보다 훨씬 작고 참여자 수·구성에 따라 동적으로 변한다.
- **Pfeifer & Ovchinnikov (2011) JIM 25(3)**: "CLV will equal WTS if (and, for the most part, only if) the
  firm's relationships with customers are **independent**."

**E1 — 우리 1차 조건도 신규가 아니다.** "한계 유지확률 × 생애가치 = 1"은 CRM 표준 FOC이고
Ovchinnikov 외가 초록에서 진술한다. 로지스틱 대입 시 `p(1−p)`가 나오는 것은 대수적 귀결이다.

**E7 — "배분 대상이 연속량뿐"이라는 방어는 성립하지 않는다.** Klein & Kolb(이산 수락/거절),
Alaei 외(2022 MS: 수익배분 + 참여 유지 + Knapsack 환원·NP-complete·PTAS), 그리고 결정적으로
**DiDi(Xu 외 KDD 2018 / Qin 외 INFORMS JAA 2020)**: "each driver-order-pair is valued in consideration of
both immediate rewards and future gains, and then dispatch is solved using a **combinatorial optimizing
algorithm**". 우리 파이프라인 골격이 실배포된 표준이다.

**E8 — 가장 위험한 최신 선행.** Kishimoto 외 (2026), *Beyond Match Maximization and Fairness:
Retention-Optimized Two-Sided Matching*, **ICLR 2026** (arXiv:2602.15752):
"we formally define the **new problem setting** of maximizing user retention in two-sided matching platforms."
**전문 11쪽에서 bundle / complement / substitut / capacity / supermodular / all-or / quota 전부 0회**를
기계 확인했다. Assumption 1의 오목성은 같은 사용자 매칭 수의 수확 체감이고, 해법(정렬 O(N log N) 환원)이
**가법 분해에 의존**하므로 구조적으로 보완형을 담을 수 없다. **금전 이전·잉여배분이 없고 단일기간이다.**

**E9 — 반응 오지정 강건성 문헌은 두껍다.** Besbes–Phillips–Zeevi(2010 M&SOM, 성과 기반 오지정 검정 —
"the ubiquitous logit model"이 통계적으로 기각돼도 성능 기준으로 통과할 수 있음),
Nambiar–Simchi-Levi–Wang(2019 MS), Cooper 외(2006 OR spiral-down), robust/DR MDP 계열.
따라서 G4의 근거를 "없다"에서 "**결정이 참여자 집합을 바꾸는 설정(오지정 + 생존 선택 결합)에 적용된 사례가
없다**"로 바꿔야 한다. 그리고 **"robust"라는 용어를 쓰면 안 되고 "sensitivity analysis"로 써야 한다.**

**E4 — 로지스틱의 직접 실증 근거는 없다.** 로짓 자체는 OM 관행이다. 관찰된 함수형은 유보값·임계형,
두 점 선형 혼합(Luy 외), 오목·포화형, 집계 도착률, 데이터 추정 생존모형이며 **오목·포화는 지지되지만
로지스틱 하반부의 볼록 구간은 지지되지 않는다.** 반례: **Aflaki & Popescu (2014 MS)** — "varying service in
the long run is not optimal", "**loyal or high-margin customers need not warrant better service**".

### 2.5 F1·F2 — 응용 맥락과 묶음 구조 ADP

**F1c 직답 — "소유하지 않은 공급자를 금전 인센티브로 유지하는 플랫폼·중개자" 동적 모형은 없다.**
가장 가까운 둘이 모두 당기 참여 유도에 그친다. **Atasoy, Schulte, Steenkamp (2020) TRR** 초록이 우리
문장과 거의 같다 — 플랫폼이 "물리적 자원을 직접 통제하지 못한다", 협업 인센티브가 "운영 수준 의사결정
모형의 필수 구성요소가 되어 동적으로 적용되어야 한다" — 그러나 협업 PDPTW MIP + 캐리어별 개별합리성
제약이고 **미래 재참여 확률·기간 간 전이·미래 가치 항이 없다**(초록 수준).
**Martínez-de-Albéniz, Pinto, Amorim (2022) M&SOM** 은 수수료 구조로 참여를 동적으로 유도하지만
**공급자가 1명**이다.

**F1b에는 강한 선례가 있다.** **Babich (2010) M&SOM** 은 동적·확률적 주기검토 DP에서 제조업체가
용량예약과 **금전 보조금**을 동시 결정하고 최적 보조금이 subsidize-up-to 구조를 가짐을 보인다.
"돈을 주어 독립 공급자 생존 확률을 올리는 DP"는 우리 것이 아니다. 방어선: 공급자 1명·단일 품목이라
**증분 가치 `c_i` 자체가 정의되지 않는다.**

**F1a — Kraljic 계보는 점수화에서 멈춘다.** 단일 공급원 위험의 정량 모형은 Kraljic을 거의 인용하지 않는
별도 계보(Tomlin 2006 MS, Babich 외 2007 M&SOM, Ang 외 2017 MS, Saghafian·Van Oyen 2016 OR)에 있다.

**F2b — 분리형 근사의 한계에 대한 최선의 인용이 확보됐다.** **Bertsimas & Popescu (Transportation
Science 2003)** 저자 사본 전문에서 추출:

> "there are two obvious drawbacks to additive bid prices: (a) They are not well defined if there are
> multiple dual solutions. (b) **They are restrictive in their way of taking into account bundles by their
> predefined additive structure.** In particular, they do not account for changes of a dual basis in response
> to accepting large-group and multileg itinerary requests."
>
> (3.2절) "they provide an additive approximation of the opportunity costs, **which are not necessarily
> additive due to bundle effects**"

개선폭은 고부하 시 가산 bid price 대비 **평균 5–10%**, 확장판 최대 20%다. 그리고 결론절의 조건 진술
(**load factor가 클 때만 차이가 나고 작으면 두 정책 모두 최적에 가깝다**)이 우리 실험 3의 조건별 설계가
선행 문헌의 표준 방식임을 보여준다. 보조 인용: Zhang(M&SOM 2011) 비분리형 min 연산자 근사, 분리형 DP
분해 대비 최대 8%(CPU +30%); Laumer–Barz(EJOR 2023) 비분리형 ALP.

**F2d — 부호 반전이 NRM에 닫힌 형태로 이미 있다.** 같은 논문 4.1절 3구간 예제:
`OC_2^LP = R1 + R23 − R13 > BP(2) = R2` (구간 하나의 기회비용이 그 구간 단독 운임 초과),
`OC_13^LP = R1 + R23 − R2 > R13`. **유보**: Proposition 3의 일반형 하한 방향은 오목성만으로도 도출되므로
보완성 고유 증거로 쓰면 과장이다 — 보완성 근거로는 예제의 닫힌 형태만 쓸 것.

**F2a — 분리형 가족은 이미 서로 동등하고 천장이 증명되어 있다.** DLP/정적 bid price(Topaloglu가
"bid price = 선형 VFA"로 명시 귀속) → RLP → 아핀(Adelman OR 2007) → 분리형 구간선형 → Lagrangian 완화
→ **동등성**: 아핀 ≡ Lagrangian, `V_PL = V_LR`(Kunnumkal–Talluri), 그리고 "Lagrangian은 가장 타이트한
분리형 구간선형 상한을 준다". 결정적으로 Topaloglu 원문에서 확인: **분리성은 정확도 판단이 아니라 저장
비용 때문에 채택된다**(`|C|^|L||T| → |C||L||T|`). 우리가 `c_i` 분해를 쓰는 동기와 동일하다.

**F2c — ATO에는 그 진술이 없다.** "분리형 VFA가 부품 보완성을 놓친다"고 한 ATO 연구는 없다.
그 진술은 NRM에서 찾아야 하고 실제로 거기 있다. (주의: Lu–Song OR 2005는 **정정 논문(OR 2019)이 존재**하므로
동시 확인 없이 인용 금지.)

**반드시 인용해야 할 새 논문**: **Dumouchelle, Frejinger, Lodi (JRPM 2024)**, *Reinforcement learning for
freight booking control problems* (arXiv:2102.00092) — 방법 구조가 우리와 가장 가깝고 **화물 도메인**이다.
지도학습으로 기간말 운영문제 최적값을 예측해 RL에 삽입하고, 상태 표현을 **선형 대 순열불변 집합**으로
비교한다(우리 실험 3의 대조와 같은 종류). 저자군이 Neur2RO·Neur2SP와 같아 **F2와 C 계보가 여기서 만난다.**
그리고 **Ma (M&SOM 2023)**: "DLP는 다품목 주문 내 품목 간 상관을 포착하지 못한다", 근사비 `q/4 → 1+ln q`
(둘 다 tight). 자원이 아니라 **주문 안 품목**을 묶는 점에서 우리와 가장 닮았고 all-or-nothing만 없다.

---

## 3. 갭 교차표

O = 있음, X = 없음. **"상호의존 종류"** 는 개체(참여자·자원) 사이 의존의 성격이다:
**대체** = 용량·수요를 두고 경쟁(증분 가치가 독립 가치보다 작아짐) / **보완** = 함께 있어야 가치가 생김
(증분 > 독립) / **혼재** = 둘이 섞임.

| 연구 | 다기간 | 참여자 집합 내생 | 배분이 결정변수 | 개별 배분 | 다품목 묶음 | 학습 VFA | 매칭을 IP로 | 비분리형을 IP에 | **상호의존 종류** | 증분>독립 명시 |
|---|---|---|---|---|---|---|---|---|---|---|
| Luy 외 2024 POM | O | O | X | X | X | X (분리형 PL) | X | X | 대체(용량) | X |
| Chen 외 2020 TR-B | O | O (도착률) | O (스칼라) | X | X | X | X | X | 대체 | X |
| Balseiro 외 2017 | O | X (IR 제약) | O | O | X | X | X | X | 해당 없음 | X |
| Klein & Kolb 2015 Omega | O | O | X (수락/거절) | O | X (단품) | X | X | X | **대체 + 시간축** | X (VIC형 하락) |
| Ovchinnikov 외 2014 MS | O | O | X (용량) | O | X | X | X | X | **대체** | X (VIC ≤ CLV) |
| Pfeifer·Ovchinnikov 2011 | O | O | X | O | X | X | X | X | **대체 + 추천(보완) 언급** | 부분 |
| Subramanian 외 2014 MS | O | O | X (유지 결정) | O | X | X | X | X | **보완(경쟁 외부성)** | **O** |
| Afèche 외 2017 M&SOM | O | O | X (용량 배급) | O | X | X | X | X | 대체 지배(§5.2 WOM은 보완) | X |
| Kishimoto 외 2026 ICLR | 부분(단일기간 LTR) | O | X (매칭만) | O | X | O | 부분 | X | **해당 없음**(오목=수확 체감) | X |
| DiDi (Xu 2018 / Qin 2020) | O | X | X | O | X | O | O | X | 대체(용량) | 미확인 |
| Bertsimas·Popescu 2003 TS | O | X (외생 용량) | X | X | **O (여정=묶음)** | X | O | X (닫힌 형태 근사) | **혼재** | **O (닫힌 형태)** |
| Ma 2023 M&SOM | O | X | X | X | **O (주문 내 품목)** | X | O | X | **혼재** | 부분 |
| van Steenbergen 외 2025 TS | O | X | X | X | X | O | O | **O** | 대체(재고) | X |
| Delarue 외 2020 NeurIPS | O | X | X | X | X | O | O | **O** | 해당 없음 | X |
| Neur2RO 2024 ICLR / Neur2BiLO | 부분(2단) | X | X | O | X | O | O | **O** | 해당 없음 | X |
| NeurADP 2020 AAAI | O | X | X | O | X | O | 부분 | X (개체별 분해) | 대체 | X |
| Babich 2010 M&SOM | O | O (공급자 생존) | **O (보조금)** | X (공급자 1명) | X | X | X | X | 해당 없음 | X |
| Atasoy 외 2020 TRR | X (당기) | X | **O (인센티브)** | O | O (협업 경로) | X | O | X | 확인 불가 | X |
| El-Neweihi 1980 MOR / Block 외 1989 | — | — | — | — | — | — | — | — | **대체·보완 부호 정리** | 형식 결과 |
| **우리 설정** | **O** | **O** | **O** | **O** | **O (all-or-nothing)** | **O** | **O** | 목표 | **혼재 (n_alt=2)** | 측정 대상 |

**교차표에서 읽어야 할 것.** 마지막 두 열이 핵심이다. 증분 > 독립을 명시한 행은 셋이고
(Subramanian 외 2014, Bertsimas·Popescu 2003, 부분적으로 Pfeifer·Ovchinnikov 2011) **메커니즘이 모두 다르다**:
경쟁 외부성 / 여정 묶음 효과 / 추천. 우리는 **다품목 all-or-nothing 조달 묶음**이라는 네 번째 메커니즘이고,
결정적으로 **"참여자 집합 내생" 열과 "묶음" 열과 "배분이 결정변수" 열이 동시에 O인 행이 우리뿐이다.**

**교차표에 불리한 열을 넣으면 우리 행이 약해진다는 것도 기록한다**(감사 지적): "상한 대비 격차"(X,
`hindsight.py` 0바이트), "최선 비교군 대비 우위"(X, 근시안을 넘지 못함), "게재 심사"(X), "이론 보증"(X).
"비분리형을 IP에" 열의 우리 셀은 **실현이 아니라 목표**다.

---

## 4. 갭 후보와 반박 대응

### 4.1 살아 있는 후보

**G1 (현상 문장).** 가산(분리형) 근사가 묶음 보완성을 놓치고 증분 가치가 독립 가치를 넘어서는 현상은
네트워크 수익관리에 이미 정식화되어 있고(Bertsimas·Popescu 2003), CRM에도 다른 메커니즘으로 있다
(Subramanian 외 2014). **우리 기여는 그 현상이 얹히는 대상이다**: 기존 문헌은 전부 **외생 용량·재고 격자**
위이고 분리형 계수의 가중치가 **상태 의존일 뿐 결정 의존이 아니다.** 우리는 **결정에 따라 전이하는 참여자
집합** 위이고 **가중치가 결정변수(재참여확률)** 다. 여기에 다품목 all-or-nothing 조달이라는 새로운 발생
원인과, 양면 참여자 모두에 대한 학습 V의 증분, 금전 잉여 몫과 품목별 배정의 단일 MILP 결합이 붙는다.
**부호는 주장이 아니라 측정 결과로 보고한다** — `n_alt = 2`에서 대체·보완이 혼재하므로 순 부호가
이론적으로 미결이다.

**G2 (측정).** 분리형 대리값 `Σ c_i p_i`의 오차를 **부호가 사전 예측된 형태로** 측정한다. 이 진단을
**학습된 가치함수 + 정수계획 계수** 맥락에서 한 연구는 없다(B4·D7·F3b에서 확인). 측정안은 §7에 적는다.

**G3 (되먹임 편향).** `c_i` 과소추정 → 잉여 덜 줌 → 이탈 → 학습 데이터 구성 축소 → 추정 악화.
spiral-down(Cooper 외 2006 OR)의 구조인데, **학습된 VFA의 오차가 자기 학습 분포를 바꾸는 참여자 유지
플랫폼**에서 이를 다룬 연구를 찾지 못했다. 저장소에 정황이 있다(공급자 c가 모든 모델에서 일관되게
과소평가, 근시안이 규칙 기반을 이김).

**G4 (반응 오지정 + 생존 선택).** 수요 반응 오지정 문헌은 두껍지만 **결정이 참여자 집합을 바꾸는 설정**에
적용된 사례가 없다. 용어는 "sensitivity analysis"로 쓴다.

**G5 (Jensen 격차).** Neur2* 계보 네 편에 학습/최적화 입력 분포 불일치 서술이 없다. 형식 고리는
Wan 외(IJCAI 2019). 단 이것은 우리가 대안 설계를 구현할 때만 유효하다.

### 4.2 죽은 갭 (주장하면 안 된다)

- "학습된 V를 MILP가 직접 최대화한다" → Delarue 외 2020, van Steenbergen 외 2025 (전문 확인 2편으로 충분).
- "비분리형 VFA를 IP에 넣는다" → 같은 논문들. 방식은 ReLU big-M 정확 임베딩.
- "결정 의존 풀링 + 헤드만 임베딩" → Neur2RO(ICLR 2024), Neur2BiLO(NeurIPS 2024).
- "분리형 근사가 상호작용에서 부정확하다" → SPAR(MOR 2004), Powell–Topaloglu 튜토리얼, Iyer 외(ICML 2013),
  **Bertsimas·Popescu(TS 2003)가 묶음에 대해 정면으로.**
- "분리형 vs 비분리형 비교" → van Steenbergen 외 2025가 Transportation Science에서 이미.
- "참여자별 증분 가치를 최적화 계수로" → VIC(MS 2014), contribution to customer equity(Omega 2015),
  DiDi(KDD 2018 / INFORMS JAA 2020).
- "유지 가치 비례 배분의 1차 조건" → CRM 표준 FOC.
- "부호 반전(증분 > 독립)이 처음" → Subramanian 외 2014, Bertsimas·Popescu 2003.

### 4.3 반박 대응표

| 반박 | 근거가 될 논문 | 우리 방어 | 통과 여부 |
|---|---|---|---|
| 증분 가치 계수는 이미 있다 | VIC, Klein & Kolb, DiDi | 인정한다. 선행 용어로 **소개**하고 기여로 주장하지 않는다 | 통과(양보) |
| 부호 반전도 이미 있다 | Subramanian 외 2014, Bertsimas·Popescu 2003 | 인정한다. 우리는 **새로운 발생 원인**과 **결정 의존 격자**다 | 조건부 통과 |
| 분리형 한계는 이미 알려졌다 | Bertsimas·Popescu, Kunnumkal–Talluri 천장 | 인정한다. **알려진 한계의 새 영역 실증**으로 쓴다 | 통과(양보) |
| 학습 VFA의 MILP 임베딩은 이미 있다 | Delarue, van Steenbergen, Neur2RO | 인정한다 | 통과(양보) |
| 이탈 내생 ADP는 이미 있다 | Luy 외 2024 POM | 결정변수가 채용 수, VFA가 분리형 PL, 반응이 금전 아님 | 부분 통과 |
| 유지 최적 매칭은 이미 있다 | Kishimoto 외 ICLR 2026 | 금전 이전 없음, 단일기간, **가법 분해 의존으로 보완형 담기 불가**(전문 기계 확인) | 통과 |
| GNN이 필요 없다 | Zhang 외 NeurIPS 2023 | **현재 우리 GNN은 검증 R² 최하위다. 예측 성능을 회복한 뒤에만 유효한 방어다** | **불통과** |
| 비분리성이 있어도 분리형이 잘 된다 | Topaloglu–Powell IJOC 2006 | Bertsimas·Popescu의 조건 진술(고부하에서만 차이) + 우리 조건별 설계 | 조건부 통과 |
| 5PL·미들마일 연구도 있다 | 5PL 개념 문헌 | 개념·역량 논의이며 최적화 모형이 아니다 | 통과 |

---

## 5. 인용 그래프에서 관찰한 단절

같은 구조의 단절이 세 곳에서 반복된다.

1. **수익배분 ↔ ADP**: Luy 외의 참조 목록에 수익배분 문헌이 한 편도 없고, 역으로 수익배분 문헌은 ADP를
   쓰지 않는다. 동적 최적화 씨앗 4편의 공통 참조는 Powell ADP 교과서와 Cachon 외 2017 단 2편이다.
2. **제약학습 ↔ ADP 분리형 근사**: 공통 인용자를 검사한 9개 조합이 **전부 0건**이다. Maragno 외 인용자
   60건을 dynamic·multiperiod·value function으로 좁히면 0건. 두 군집을 잇는 경로는
   Anderson 외 2020 정식화 → Delarue 외 2020 → van Steenbergen 외 2025 뿐이다.
3. **NRM 묶음 효과 ↔ 유지·잉여배분**: Bertsimas·Popescu(185편)와 Zhang(50편)의 피인용 교집합 15편이
   전부 고전 NRM이고, 학습 VFA·유지·이탈·잉여배분을 다룬 논문이 **0편**이다(제목 수준 관찰).
4. 집합함수 semigradient 문헌은 어느 군집과도 공통 인용자가 0이다. 신뢰도 이론의 joint importance
   계보도 ADP와 만나지 않는다(Birnbaum × ADP 가치함수 = 0건).

---

## 6. 확인하지 못한 것 (도서관 접속 우선순위)

| 순위 | 항목 | 왜 중요한가 |
|---|---|---|
| 1 | **Klein & Kolb (2015) Omega 55:111–125** | "contribution to customer equity"를 **최적화 계수로 쓰는가 사후 지표로만 쓰는가.** 전자면 우리 증분 가치 계수의 신규성이 더 약해진다. Crossref·RePEc·Unpaywall·OpenAlex 네 경로 모두 본문 미제공 |
| 2 | **Subramanian, Raju, Zhang (2014) MS 60(2):494–507** | 증분 > 독립의 명제 형태와 메커니즘. G1의 양보 범위를 정한다. 초록만 읽음 |
| 3 | **El-Neweihi (1980) MOR 5(4):553** + **Block·Griffith·Savits (1989) AAP 21:919–929** | 부호 논증의 최강 인용인데 **초록만** 읽었다. r=2 특성화의 정확한 진술과 가정 확인 필수 |
| 4 | **Bertsimas & Popescu (2003) TS** 조판 원문 | Proposition 3의 정확한 부등호 방향(PDF 추출본에서 어긋나 보임). 논증은 §4.1 예제에 의존하므로 치명적이지 않으나 인용 전 확인 |
| 5 | **Laumer & Barz (EJOR 2023)** | F2b의 최신 핵. 비분리형 ALP의 개선폭과 계산 비용 |
| 6 | **Hildebrandt 외 (Networks 2026)** | 오픈액세스이므로 확보 가능. MILP 정식화가 big-M인가, 망 크기, 해 시간, 가치함수 오차 착취 보고 여부 |
| 7 | **NeurADP (AAAI 2020)** 전문 | 개체별 분해 오차를 어디서든 정량화했는가. G2 선행성 판정에 직결 |
| 8 | **Dumouchelle, Frejinger, Lodi (JRPM 2024)** 결과 표 | 선형 대 순열불변 집합 표현의 승패. 우리 실험 3 대조의 선례 |
| 9 | **Luy 외 (2024) POM** | 완전정보 lookahead 벤치마크의 정의와 가려진 수치, PL-VFA 분리 가능성 가정의 위치 |
| 10 | **Topaloglu–Powell (IJOC 2006)** | 분리형 근사가 모든 대체 패턴에서 잘 작동한 이유에 대한 저자 설명 |
| 11 | Zhu & Kuo (2012 Annals of OR) "Importance measures in reliability and mathematical programming" | 초록조차 미확보. F3b "없다" 단정의 유일한 잔여 구멍 |
| 12 | MS 2025 "Retention or Acquisition? Behavior-Based Quality Disclosure" | 초록이 어디에도 등록되지 않아 선별 불가 |
| 13 | Chen 외 (2020) TR-B, Musalem 외 (2023) OR, Masorgo 외 (2026) JOM, Shi 외 (2024 IJOC), Song (1998) OR, Pulles 외 (2023 JoM) | 각각 §2의 특정 주장을 보강하거나 모델 가정에 반증을 준다 |

**공통 과제**: 저널 사분위 전수 확인, OpenAlex 전방 피인용 순회 재시도(UTC 자정 이후 또는 무료 API 키).

---

## 7. 시뮬레이터·실험 설계에 대한 시사점

**1. 2차 교차차분 측정을 최우선으로 한다 (부호가 사전 예측된 유일한 측정).**
`coef.py`에 2명 제거를 추가해
`Δ_ij V = V(S) − V(S∖i) − V(S∖j) + V(S∖{i,j})`
를 재고, 공급자 쌍을 **(A) 같은 품목 대체 공급자 쌍** 대 **(B) 품목이 겹치지 않는 쌍**으로 나눈다.
이론 예측은 **A ≤ 0, B ≥ 0**이다(신뢰도 joint importance 부호 + Heo 외의 중복/보완 쌍 부호).
공급자 쌍 15개, MILP 불변, 예측 1회. 이 측정이 "유지 가치 변환에서 구조 정보가 소실되었는가"에 직접 답하고,
G1의 부호를 주장이 아니라 결과로 만든다. 단조성 전제는 §2.3에서 이미 확인됐다.
`items_per_order = 1` 대조 조건에서는 보완이 사라지므로 **B의 양의 부호도 사라져야 한다** — 기존
대조 조건보다 예리한 예측이다.

**2. 후견 상한을 채운다. 다만 근거를 정확히 쓴다.**
빈 `src/match/hindsight.py`를 채우고 결과 표에 상한 대비 격차를 싣는다. 단 감사 지적을 반영한다:
인용 근거인 You·Vossen과 Aouad·Sarıtaç는 **외생 이탈** 모형이고 우리는 전이확률이 결정 의존이므로
ALP 완화 상한이 직접 이식되지 않는다. **완전정보 lookahead**(Luy 외의 방식)가 더 안전한 경로다.
그리고 "근시안 우위는 상한 없이 해석 불가"는 틀렸다 — 상한은 순서의 원인을 설명하지 않는다.

**3. 실험 3을 다시 돌리기 전에 공정성 장치를 먼저 갖춘다.**
현재 상태로는 "공정 비교"라고 쓸 수 없다(§2.2 C3). v2 처방에 **외부 CV 또는 시드 반복**과
**GNN 입력 종류별 표준화**를 추가해야 하고, 그 전의 v1 수치는 "구조 정보 무효"가 아니라
**"구현 결함이 섞인 미결"** 로 보고한다. 그리고 GNN 예측 성능(네 칸 R² 최하위, 한 칸 0.007)을 회복하기
전에는 GNN을 기여로 내세우지 않는다.

**4. 조건별 설계의 근거를 Bertsimas·Popescu로 명시한다.**
"차이가 모든 조건에서 나지 않고 특정 조건에서만 난다"는 것은 결함이 아니라 **NRM의 표준 결과 형태**다
(고부하에서만 5–10%, 저부하에서는 두 정책 모두 최적에 가까움). 우리 공급 편중도·묶음 크기 축이 같은 역할이다.

**5. 분리성의 동기를 저장 비용으로 정직하게 쓴다.**
Topaloglu가 분리성을 **정확도가 아니라 저장 비용** 때문에 택한다고 적는다(`|C|^|L||T| → |C||L||T|`).
우리 `c_i` 분해도 같은 이유이며, 이것이 "왜 분리형을 쓰는가"에 대한 가장 정직한 답이다.

**6. 용어를 확립된 것으로 바꾼다.**
`c_i` = **discrete supergradient** = **다선형 확장의 p=1 1차 Taylor** = **Birnbaum 중요도(p=1 평가)**.
구조는 **parallel-in-series**. 같은 품목 쌍은 **reliability substitutes**, 다른 품목 쌍은
**reliability complements**. **"초모듈"을 무조건 쓰면 정리와 충돌한다**(이진 한정 금지).
강건성 실험에는 "robust"를 쓰지 않고 **"sensitivity analysis"** 를 쓴다.

**7. 실험 2의 우선순위를 올리고 되먹임 편향을 측정 항목에 넣는다.**
G4가 살아 있으므로 보류를 유지할 이유가 약하다. 그리고 G3(되먹임 편향)은 새 측정 항목이 필요하다:
기간별로 `c_i` 추정 오차와 학습 데이터 구성(참여자 수·유형 분포)을 함께 기록해 spiral-down 여부를 본다.

**8. 모델 가정에 대한 반증 후보를 한계 절에 적는다.**
Aflaki & Popescu (2014 MS)는 장기적으로 서비스를 차등하는 것이 최적이 아니라고 하며 "loyal or high-margin
customers need not warrant better service"라고 적는다 — `c_i` 비례 차등 배분에 대한 심사지 반례다.
Pulles 외 (2023 JoM)는 의존이 관계 특수 투자의 효과를 음의 방향으로 조절한다고 보고한다 —
로지스틱 **단조 증가** 가정에 대한 실증 반증 후보다(전문 확인 필요). 로지스틱의 직접 실증 근거는 없고,
문헌이 지지하는 것은 **오목·포화**까지다.

**9. 가드를 신뢰영역 관점에서 다시 본다.**
`coefs()`의 shrink + 절단은 Delarue 외의 하한 감싸기에 가까운 **사후 절단**이며 입력공간 신뢰영역이 아니다.
OptiCL·Shi 외(IJOC 2024)의 신뢰영역과의 격차를 한계로 명시하고, 공급자 c 과소평가와 가드의 상호작용을
점검한다(경쟁 가설 넷 중 하나다).

**10. 표본 불균형을 보고한다.**
공급자 표본이 42~48개, 주문자가 349~381개다. 공급자 c 과소평가의 경쟁 가설 중 하나이므로
참여자 단위 CI와 함께 명시해야 한다.
