# 선행연구 조사 범위 (1차: 주제 A·B)

목적: 초록의 범위(다품목 원자재 B2B 플랫폼, 매칭·잉여배분, 재참여 내생성, GNN 가치함수 근사 + MILP 결합)
안에서 **리서치 갭을 특정**한다. 문헌 요약이 목적이 아니라, "아무도 채우지 않은 칸"을 찾는 것이 목적이다.

1차 조사는 주제 A·B만 한다. C(GNN VFA의 노드 삭제 차분 한계), D(묶음 보완성/초모듈성),
E(잉여배분 몫의 특성화)는 2차로 미룬다. A·B 결과를 보고 2차 범위를 다시 정한다.

기간: 2015년 이후를 주로 보되, 방법의 원전(ADP 근사구조, 제약학습)은 연도 제한 없이 추적한다.
출처 기준: SCIE/SSCI Q1 우선(사분위 확인), OR/OM 주요 저널(MSOM, Management Science, Operations Research,
POM, Transportation Science, EJOR, IJPE)과 ML 주요 학회(NeurIPS, ICML, ICLR, AAAI)를 우선한다.
페이월 전문은 접근하지 못할 수 있다 → 그런 항목은 "직접 확인 필요"로 표시하고 확인할 질문을 적는다.

---

## 주제 A: 내생적 참여자 구성 (플랫폼의 결정이 미래 참여자 집합을 바꾸는 동태)

우리 설정: 매 기간 플랫폼이 (1) 주문 수락 (2) 품목별 공급자·수량 배정 (3) 주문자·공급자·플랫폼 간
거래잉여 배분을 정한다. 참여자의 다음 기간 재참여 확률이 그 기간 배분받은 잉여에 의존한다.
따라서 상태(참여자 집합과 그 거래 연결 구조)가 결정 의존적으로 전이된다.

핵심 질문
- A1. 플랫폼의 당기 결정이 **다음 기간 참여자 집합**을 바꾸는 것을 명시적으로 모형화한 동적 최적화 연구는 무엇이 있는가?
      (차량공유 기사 유지·재배치, 긱/프리랜스 노동 플랫폼 공급 이탈, 양면 시장 churn, 크라우드소싱 물류)
- A2. 그 연구들에서 **결정변수가 무엇인가**: 가격/수수료인가, 매칭인가, 아니면 **잉여배분(수익 공유 몫)** 인가?
      잉여배분을 기간별 결정변수로 둔 동적 모형이 존재하는가?
- A3. 참여자 이탈 반응을 어떤 함수로 두는가(로지스틱, 임계값, 학습된 반응). 그 함수의 추정·강건성을 다룬 연구가 있는가?
- A4. 미들마일/B2B 원자재 거래, 5PL, 다품목 묶음 조달 맥락에서 이 동태를 다룬 연구가 있는가? (없을 것으로 예상 — 확인 필요)
- A5. 이 문헌들이 쓰는 성과 지표와 벤치마크는 무엇인가(근시안, 규칙 기반, 후견 상한, 유체 극한).
      우리가 쓰는 비교군이 이 문헌의 표준에 비해 약한 부분이 있는가?

검색 각도 (키워드는 조합해서 변형)
- endogenous supply / participant churn / driver retention / seller attrition + dynamic matching
- revenue sharing + platform + dynamic program / retention
- two-sided platform + dynamic pricing + participation decision
- state-dependent arrival rate / decision-dependent uncertainty + MDP
- freight matching platform / middle mile / less-than-truckload marketplace + dynamic
- carrier retention + freight brokerage + dynamic assignment

## 주제 B: 학습된 가치함수를 MILP에 임베딩하는 방법 (분해 없는 결합)

우리 현재 구현: V를 학습한 뒤 참여자별 유지 가치 c_i = V(s) − V(s에서 i 제외)로 **1차 분해**하고,
MILP 목적함수에 Σ c_i × PWL(재참여확률_i)로 참여자별 분리형으로 넣는다 (src/match/milp_build.py).
이 분해 단계에서 보완성(여러 공급자가 동시에 남아야 묶음 주문이 성사되는 효과)이 소실된다.

핵심 질문
- B1. 학습된 신경망을 수학적 최적화 모형의 목적함수·제약에 **정확히 임베딩**하는 방법론의 현재 표준은 무엇인가?
      (constraint learning, OptiCL, JANOS, OMLT, gurobi-machinelearning, ReLU big-M/indicator 정식화)
      계산 한계는 어디까지 보고되는가(은닉층 폭·깊이, 해 시간).
- B2. 그 임베딩을 **주기별로 반복 해결하는 ADP/rollout 구조**에 쓴 연구가 있는가?
      즉 "학습된 V를 사후 상태(post-decision state) 가치로 두고 MILP가 직접 최대화"하는 사례.
- B3. ADP 문헌에서 가치함수 근사를 최적화에 넣기 위해 쓰는 **표준 근사 구조**는 무엇인가:
      분리 가능 오목/구간선형(Powell, Topaloglu 계열 자원배분), 선형 기저함수, 아핀 완화.
      **비분리형(nonseparable) VFA를 정수계획에 넣은 연구**가 있는가? 있으면 어떻게 다루는가?
- B4. leave-one-out 방식의 노드/자원 삭제 차분으로 선형 계수를 뽑는 근사(우리 방식)에 해당하는
      기존 용어와 선행 사례가 있는가? (marginal value, gradient-based VFA, opportunity cost approximation)
      그 근사가 보완성·초모듈성이 있을 때 편향된다는 지적이 이미 있는가?
- B5. GNN 임베딩을 최적화 모형에 결합한 사례가 있는가? 그래프 구조가 결정과 무관하고
      노드 생존 확률만 결정 의존일 때의 선형 읽기(readout) 구조를 쓴 선례가 있는가?
- B6. 학습된 목적함수를 최적화에 넣을 때 보고되는 실패 양식은 무엇인가
      (분포 이동, 최적화가 모델 오차를 착취하는 현상, 신뢰 영역·가드 장치).

검색 각도
- optimization with constraint learning / embedding trained neural network in MILP
- ReLU network big-M formulation / mixed integer formulation of neural networks
- surrogate objective learned by neural network + integer programming
- approximate dynamic programming + separable concave value function approximation + resource allocation
- post-decision state value function + integer program + rollout
- graph neural network + combinatorial optimization + value function (branch: learning to optimize)
- predict-then-optimize / decision-focused learning (인접 문헌: 목적 학습과 최적화 결합)

---

## 산출물에서 반드시 답해야 할 것

1. A·B 각각에서 **가장 가까운 선행연구 3~5편**을 특정하고, 우리 설정과 무엇이 다른지 한 문장씩 적는다.
2. 갭 교차표: 행 = 선행연구, 열 = (동적 / 참여자 집합 내생 / 잉여배분 결정변수 / 다품목 묶음 /
   학습 VFA / 비분리형 VFA를 IP에 임베딩). 빈 칸이 갭 후보다.
3. B4의 답에 따라 우리 음의 결과(1차 분해에서 구조 정보 소실)의 위치를 정한다:
   기존에 지적된 한계의 실증인가, 처음 보고하는 것인가.
4. 갭 주장별로 **반박 가능성**을 적는다(누가 "이미 있다"고 말할 수 있는지, 어떤 논문을 근거로).
5. 페이월로 확인 못 한 항목 목록과, 도서관 접속으로 확인할 구체적 질문.

---

# 2차 조사 범위 (주제 C·D·E)

착수: 2026-09-28. 1차(A·B) 결과를 반영해 **중복을 제거하고 다시 정의했다.**
1차에서 이미 답이 나온 것은 다시 묻지 않는다:
- 노드 삭제 1차 차분의 기존 용어(discrete semigradient)와 그 편향이 이미 지적되었는지 → **B4에서 답 나옴**
- 학습 신경망을 MILP에 정확 임베딩하는 표준과 계산 한계 → **B1에서 답 나옴**
- 잉여배분 몫을 기간별 결정변수로 둔 동적 모형의 존재 → **A2에서 답 나옴**

원자료: `ref/_review/endogenous-participation-vfa-embedding_research_A.md`, `..._research_B.md`,
종합: `..._cited.md`. 2차 조사자는 이 세 파일을 먼저 읽고 중복을 피한다.

## 주제 C: 구조 정보가 요약 지표를 이기는 조건 (실험 3의 선행 근거)

우리 실험 3은 "시장 요약 지표만 쓰는 MLP vs 거래 연결 구조를 쓰는 GNN"을 공급 편중도 {0, 0.5, 1} ×
주문당 품목 수 {1, 2, 3}에서 비교한다. 대조 조건은 주문당 품목 수 1이다.

- C1. OR 응용(재고, 차량 라우팅, 수익관리, 동적 매칭)에서 **GNN을 가치함수 근사로 쓴 사례**는 무엇이며,
      요약 지표 기반 근사(MLP·선형)와 비교했을 때 **얼마나, 어떤 조건에서** 나았다고 보고하는가?
- C2. "구조 정보가 요약 통계보다 유용해지는 조건"을 **명시적으로 실험 설계에 넣어** 검증한 연구가 있는가?
      (그래프 밀도, 편중도, 이질성, 병목 등을 축으로 둔 ablation)
- C3. GNN과 MLP를 **공정하게 비교하는 프로토콜**의 표준은 무엇인가? (파라미터 수 맞추기, 학습 예산 동일,
      동일 특징 정보량 보장, 랜덤 시드 반복 수) 우리 방식(같은 튜닝 예산, 같은 시장 요약 지표)에 빠진 것이 있는가?
- C4. 그래프 신경망 표현이 **가변 노드 수·활성 마스크**를 다루는 표준 처리와, 그 처리가 예측 성능에 주는 영향을
      보고한 연구가 있는가?
- C5. 반대 증거: "이 규모·이 문제에서는 GNN이 MLP를 이기지 못했다"고 **음의 결과를 보고한** 논문이 있는가?
      (우리 실험 3 v1 결과의 선례이자 방어 근거)

검색 각도: graph neural network value function approximation inventory routing revenue management;
GNN vs MLP ablation graph structure information gain; learning to optimize combinatorial graph representation
negative result; permutation invariant set encoder variable number of agents; deep sets vs GNN comparison.

## 주제 D: 묶음 보완성과 가치함수의 초모듈성 (G3의 메커니즘 근거)

우리 주장: 다품목 all-or-nothing 주문이 있으면 여러 공급자가 **동시에** 남아야 주문이 성사되므로
미래 가치가 참여자 집합에 대해 **초모듈적**이고, 따라서 1차 분해(모듈러 근사)가 체계적으로 과소평가한다.
이 주장의 형식적 근거와 선례를 찾는다.

- D1. 다품목 **all-or-nothing** 주문(묶음 주문, 완전 충족 조건)을 동적 자원배분·재고 모형에서 다룬 연구.
      assemble-to-order(ATO) 문헌이 가장 가까운 후보다: 부품 공통성·동시 가용성이 가치함수에 주는 구조.
- D2. 가치함수의 **초모듈성·열성(supermodularity/submodularity)** 을 형식적으로 진술하고 활용한
      동적 자원배분 연구. 특히 "보완재가 있으면 가치함수가 초모듈"이라는 진술의 출처.
- D3. 초모듈·열성 집합함수를 **선형(모듈러)으로 근사할 때의 오차 한계** 문헌 (curvature, correlation gap,
      Iyer 외 계열의 근사 보증). B4에서 semigradient의 tight성은 확인했으니, 여기서는 **오차 크기의 경계**를 찾는다.
- D4. 조합 경매·묶음 입찰(combinatorial auction, package bidding)에서 보완성이 있을 때
      개별 항목 가격(선형 가격)이 실패한다는 결과. 우리 1차 분해 실패와 구조적으로 같은 현상인가?
      (선형 가격의 비존재 = 모듈러 근사의 한계와 같은 뿌리인지 확인)
- D5. 다품목 조달에서 **공급자 포트폴리오의 옵션 가치**(단독 공급 품목의 위험, 이중 소싱)를 동적으로 다룬 연구.
      우리가 검토한 "미래 주문 성사 가능성" 타깃의 선행 개념이 있는가?

검색 각도: assemble-to-order common components dynamic allocation value function;
supermodular value function dynamic resource allocation complementarity;
submodular set function modular approximation bound curvature correlation gap;
combinatorial auction complementarity nonexistence of linear prices package bidding;
dual sourcing supplier portfolio option value multi-item procurement dynamic.

## 주제 E: 배분 몫의 특성화와 생애가치 기반 배분 (후보 3의 근거)

검토 중인 분석 결과: 로지스틱 반응 + 유지 가치 c_i에서 최적 잉여 몫이
`b·c_i·p(1−p)/w = 1` 형태의 1차 조건으로 특성화된다. 이 결과의 선행성과 인접 문헌을 찾는다.

- E1. 잉여·수익 배분 몫의 최적 수준을 **참여 반응의 1차 조건으로 특성화**한 결과가 있는가?
      (플랫폼 수수료 설계, 수익공유 계약, Nash bargaining, 협상해의 비교정태)
- E2. **가장 중요**. 마케팅·수익관리의 **고객 생애가치(CLV) 기반 동적 자원배분** 문헌.
      "고객별 생애가치를 추정해 그에 비례해 마케팅·할인·서비스 자원을 배분한다"는 구조는 우리 c_i와 직접 대응한다.
      이 문헌에 (a) 생애가치를 학습으로 근사하고 (b) 그것을 최적화 모형의 계수로 넣는 선례가 있는가?
      **있으면 우리 c_i 개념의 신규성이 약해지므로 반드시 확인해야 한다.**
- E3. 개별 참여자에게 **차등 배분**하는 것의 공정성·전략적 문제를 다룬 연구.
      (차등 수수료의 공정성, 참여자가 배분 규칙을 알고 행동을 바꾸는 유인 문제)
      우리 모델은 참여자를 전략적으로 두지 않으므로, 이 한계를 어떻게 서술해야 하는지의 근거.
- E4. 유지·이탈 반응이 **금전 배분에 로지스틱**으로 반응한다는 가정의 실증적 근거 또는 반증.
      (A3에서 로지스틱 선례를 찾지 못했으므로, 인접 분야 — 고객 이탈 예측, 가격 반응 — 에서 찾는다)
- E5. 배분 규칙을 **고정 비율로 두는 것**이 실무 표준인지에 대한 근거 (우리 규칙 기반 비교군의 정당화).

검색 각도: customer lifetime value based resource allocation optimization; CLV estimation machine learning
marketing budget allocation integer program; optimal revenue sharing rate first order condition platform;
Nash bargaining surplus division operations platform; differential commission fairness marketplace;
churn response to monetary incentive logistic elasticity.

## 2차 산출물에서 반드시 답해야 할 것

1. C5·D3·E2 각각에 대한 직답. 특히 **E2에서 CLV 문헌에 선례가 있으면 그것을 명시하고 우리 차이를 적는다.**
2. 1차의 갭 후보 G1~G5를 **강화하거나 약화하는** 증거를 분류한다.
3. D에서 "묶음 보완성 → 가치함수 초모듈성 → 모듈러 근사의 체계적 과소평가"라는 우리 논증 사슬의
   각 고리에 인용할 수 있는 문헌을 지정한다. 인용할 수 없는 고리는 "우리가 직접 논증해야 하는 부분"으로 표시한다.
4. 페이월로 확인 못 한 항목과 확인할 질문.

## 2차 조사 필수 추가 (reviewer 적대적 감사 결과 반영, 2026-09-28)

1차 초안의 갭 후보 G2·G3가 감사에서 무너졌다 (`ref/_review/..._review.md`). 2차 조사는 아래를 **반드시** 포함한다.

### C에 추가: Neur2RO 계보 (G2 선례)
Neur2RO (Dumouchelle, Julien, Kurtz, Khalil, ICLR 2024, arXiv:2310.04345)가 우리 G2와 같은 구조를 이미 한다:
원소별 임베딩이 결정변수의 함수 → 집계(풀링) → 헤드 신경망 → 결정 의존 부분만 MILP에 임베딩하고
결정 무관 부분은 사전 계산 (gurobi-machinelearning 1.3.0). 이 인용은 직접 확인했다.
- C6. 이 계보(Neur2SP NeurIPS 2022, Neur2BiLO, Wang 외 IJOC 2023, 그리고 이들의 피인용)에서
      **풀링 가중치가 확률·비율 같은 연속 결정변수**인 사례가 어디까지 있는가?
      우리와 남는 차이("가중치가 구간선형 재참여확률", "참여자 집합이 결정에 따라 전이하는 ADP 루프")가
      이 계보 안에서 이미 다뤄졌는가?
- C7. 이 계보가 **V(기대 입력) vs 기대 V(입력)의 Jensen 격차**를 어떻게 다루는가?
      학습 시 입력 분포와 최적화 시 입력 분포가 다른 문제(우리: 학습은 p∈{0,1} 실현 그래프, 최적화는 p∈(0,1))를
      인지·처리한 연구가 있는가? 이것이 G2의 알려지지 않은 위험이다.

### D에 추가: 초모듈성 부호 문제 (G3 개념 오류)
감사 지적: 우리 관측(공급자 c 과소평가)의 부호가 초모듈성 논증과 반대다. 초모듈성은 leave-one-out의
**과대평가**를 낳는다. 또 n_alt = 2에서 같은 품목의 두 공급자는 **대체재**이고 다른 품목 공급자는 보완재여서
순 부호가 미결이다.
- D6. **가장 중요**. 자원이 대체재와 보완재를 **동시에** 포함할 때 가치함수의 초모듈·열성 부호를 판정한 연구.
      "일부는 대체, 일부는 보완"인 집합함수의 모듈러 근사 편향 방향에 대한 형식적 결과가 있는가?
- D7. leave-one-out(한 개체 제거) 차분이 **다수 개체 동시 이탈**을 예측할 때의 오차를 다룬 연구.
      우리 c_i는 1명 제거만 계산하는데 실제로는 여러 명이 동시에 떠난다.

### E에 승격: CLV/CRM 동적계획 (G1을 위협)
감사 지적: "돈을 주면 유지확률이 오르고 미래가치가 현재 결정에 들어온다"는 CRM/CLV 동적계획의 **표준 정식화**이며
1차 조사에 전혀 없었다. 단서: Lewis (2005) *Management Science*, Pfeifer & Carraway (2000).
- E2를 최우선으로 올린다. 추가 질문:
  - E6. CLV 기반 동적 자원배분에서 **고객별 생애가치를 최적화 모형의 계수로 넣은** 선례가 있는가?
        있으면 우리 c_i의 신규성이 사라진다. 있는지 없는지 단정적으로 답하라.
  - E7. 그 문헌이 **매칭·용량 배분 같은 조합 결정**과 결합된 사례가 있는가, 아니면 배분 대상이
        마케팅 예산·할인 같은 연속량뿐인가? 후자라면 우리 차별점이 거기에 있다.
  - E8. 유지 최적화를 매칭과 결합한 최근 연구 (감사가 지적한 arXiv 2602.15752, ICLR 2026 계열 포함).

### E에 추가: 오지정 강건성 인접 문헌 (G4)
감사 지적: G4의 근거가 "못 찾았다"뿐이다. 수요 반응 모형 오지정 강건성 문헌이 조사에 없었다.
단서: Nambiar, Simchi-Levi, Wang (2019) *Management Science*.
- E9. 수요·행동 반응 모형의 **오지정 강건성**을 다룬 운영 문헌의 표준 접근(모형 오지정 하의 가격결정,
      robust MDP, 분포적 강건 최적화)은 무엇인가? 우리 실험 2 설계(재학습 / 불일치 평가)가
      그 표준에 비해 무엇이 빠져 있는가?

---

# 3차 조사 범위 (주제 F) — 상호의존의 종류를 축으로

착수: 2026-09-28. 1·2차 조사(A~E) 결과 우리 핵심 개념 `c_i`의 선행이 확인됐다
(Klein & Kolb 2015 Omega의 contribution to customer equity, Ovchinnikov 외 2014 MS의 VIC,
Pfeifer & Ovchinnikov 2011 JIM). 그러나 **그 선례들의 상호의존은 용량 경쟁, 즉 대체형**으로 보인다.
용량이 제한되면 VIC < CLV, 즉 **증분 가치가 독립 가치보다 작아진다.**

우리 설정은 반대다. 다품목 all-or-nothing 묶음에서 어떤 품목의 유일한 공급자는 그 품목뿐 아니라
**묶음 전체의 성립을 가능하게** 하므로 증분 가치가 독립 가치를 **넘어선다** (보완형 = 초모듈 방향).

**따라서 3차 조사의 축은 "상호의존의 종류(대체 / 보완 / 혼재)"다.** 모든 F 조사자는 찾은 문헌을
이 축으로 분류해 보고한다. 최종 리뷰의 갭 교차표에도 이 열을 추가한다.

## G1 재서술 (요소 나열 → 현상 문장)

기존(요소 나열, 리뷰어가 "그 차이가 왜 중요한가"로 깨뜨림):
"개별 잉여 몫이 결정변수 + 확률적 재참여 + 참여자 집합 전이 + 다품목 묶음 + 학습 VFA를 MILP에"

새 형태(현상 문장):
**"다품목 묶음 주문이 만드는 보완형 상호의존 때문에, 참여자의 증분 가치가 그의 독립 가치를 넘어선다.
용량 경쟁이 만드는 대체형 상호의존에서는 증분 가치가 독립 가치보다 작아지므로(VIC < CLV),
기존 선례의 부호와 반대다. 이 부호 반전이 잉여배분 결정을 질적으로 다르게 만든다."**

이 문장은 검증 가능하다: `V(S) − V(S∖{i})` 대 `V({i}) − V(∅)`를 재면 된다.

## F1: 구매 관리·공급망 위험

- F1a. 병목 품목·공급 위험 분류(Kraljic 포트폴리오 등)에서 **단일 공급원 위험**을 정량 모형으로 다룬 연구.
- F1b. **핵심 공급자 유지 인센티브**: 구매자가 공급자를 붙잡기 위해 금전(가격 프리미엄, 물량 보장, 조기 지불)을
      쓰는 것을 최적화 모형으로 다룬 연구.
- F1c. **가장 중요**. "**소유하지 않은 공급자를 금전 인센티브로 유지하는 플랫폼·중개자**" 모형이 있는가?
      우리는 플랫폼이 공급자를 고용하지 않고 잉여 배분으로만 붙잡는다. 화물 중개(freight brokerage),
      B2B 마켓플레이스, 조달 대행에서 이런 모형이 있는지 확인하라. 없으면 없다고 단정하라.
- F1d. 이 문헌들의 상호의존은 대체형인가 보완형인가 혼재인가.

## F2: 묶음 구조의 ADP

- F2a. **네트워크 수익관리**에서 자원(구간) 간 보완 관계를 가치함수 근사가 어떻게 다뤘는가.
      다구간 여정(itinerary)은 여러 자원을 동시에 소비하므로 우리 묶음과 구조가 같다.
      DLP·RLP·분해(decomposition) 근사, 아핀·분리형 근사의 계보.
- F2b. **분리형 근사의 한계를 명시적으로 지적한 연구**. 특히 "여정이 여러 구간을 쓰므로 분리형 근사가
      보완성을 놓친다"는 진술과, 그것을 개선한 근사(아핀, 분해, Lagrangian).
      B 조사에서 Powell–Topaloglu 튜토리얼과 SPAR는 이미 확보했으니 **네트워크 RM 쪽 계보**를 찾는다.
- F2c. 조립형 생산(ATO)에서 부품 간 보완을 가치함수 근사가 다룬 방식. D 조사에서 Chen–Long–Qi(OR 2021)와
      Song(1998)은 확보했으니 **근사 방법론** 쪽을 본다.
- F2d. 이 문헌들이 우리 부호 반전(증분 > 독립)을 명시적으로 말한 곳이 있는가.

## F3: 구조 기반 가치 근사

- F3a. **직렬-병렬 신뢰도 구조**를 가치함수 근사나 자원 평가에 쓴 선례.
      우리 묶음은 품목에 대해 직렬(모든 품목 필요)이고 품목 내 공급자에 대해 병렬(하나면 충족)이다.
      이 구조가 신뢰도 이론의 직렬-병렬 시스템과 정확히 대응한다.
- F3b. **Birnbaum 중요도**(또는 구조 중요도, Shapley-Birnbaum)를 공급자·자원 평가에 쓴 선례.
      Birnbaum 중요도는 정확히 "그 요소가 작동/비작동일 때 시스템 신뢰도의 차"이므로 우리 `c_i`의
      신뢰도 이론 대응물이다. 이것이 ADP 가치함수나 공급자 선택·유지에 쓰인 사례가 있는가.
- F3c. 신뢰도 중요도 측정치가 **다수 요소 동시 고장**에서 어긋난다는 지적(공동 중요도, joint importance).
      D7(다수 동시 이탈)의 신뢰도 이론 대응물이다.
- F3d. 이 계보가 우리에게 주는 것: `c_i`를 Birnbaum 중요도로 재해석할 수 있으면 부호 논증이
      신뢰도 이론의 기존 결과로 뒷받침되는가?

## F4: 선행 확인 (가장 결정적)

- F4a. **Afèche, Araghi, Baron (2017) M&SOM**의 "policy-dependent value"에서 개체 간 상호의존이
      **대체형(용량 경쟁)뿐인가, 보완형을 포함하는가.**
- F4b. **Klein & Kolb (2015) Omega 55:111–125**의 "contribution to customer equity"에서
      (i) 그것을 **최적화 계수로 쓰는가 사후 지표로만 쓰는가**, (ii) 상호의존이 **대체형뿐인가**.
- F4c. **Ovchinnikov, Boulu-Reshef, Pfeifer (2014) MS**의 VIC에서 용량 제한이 VIC를 CLV보다 **작게** 만든다면,
      **VIC > CLV가 되는 경우(보완형)를 그 논문이나 후속 문헌이 다루는가.**
- F4d. **Pfeifer & Ovchinnikov (2011) JIM**의 "independent" 조건이 깨지는 방식으로 **보완**을 언급하는가,
      아니면 경쟁·잠식(cannibalization)만 다루는가.
- F4e. **Kishimoto 외 (2026) ICLR** *Retention-Optimized Two-Sided Matching*의 유지 곡선이 매칭 수에 **오목**이라면
      그것은 수확 체감이고 보완형이 아니다. 이 논문에 **묶음·보완형 상호의존이 전혀 없다는 것을 확인하라.**

## 3차 산출물에서 반드시 답해야 할 것

1. F1c와 F3b에 대한 직답(선례 유무).
2. **F4 전체에 대한 직답**: 확인된 선례들의 상호의존이 대체형뿐이라면, G1의 현상 문장(부호 반전)이 성립한다.
   보완형을 이미 다룬 선례가 있으면 그것을 명시하고 G1을 다시 좁혀야 한다.
3. 찾은 문헌 전부를 **상호의존 종류(대체 / 보완 / 혼재 / 해당 없음)** 로 분류한 표.
4. 페이월로 확인 못 한 항목과 확인할 질문.
