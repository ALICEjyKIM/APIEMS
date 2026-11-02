# 주제 B 증거 수집: 학습된 가치함수를 MILP에 임베딩하는 방법

조사일: 2026-09-28 · 담당: researcher 서브에이전트
범위 문서: `ref/literature-review.md` 주제 B (B1–B6) · 계획: `ref/_review/endogenous-participation-vfa-embedding_plan.md`
목적: 문헌 요약이 아니라 **리서치 갭 특정**. 기준선은 우리 현재 구현(`src/match/milp_build.py`, `src/vfa/coef.py`).
검색 경로: WebSearch + OpenAlex API 인용 그래프 순회(키 불필요) + arXiv/ar5iv 전문 + 저자 페이지 PDF 직접 추출(pypdf).

---

## 0. 우리 방식의 정확한 기술 (갭을 재는 기준선, 코드 직접 확인)

- `vfa/coef.py: raw()` — 학습된 가치 모델 V에 대해 c_i = V(phi(o)) - V(phi(o에서 i 제외)).
  주문자·공급자 각각 노드 1개씩 제거한 입력을 배치로 넣어 예측값 차분을 취한다.
  즉 **leave-one-out 노드 삭제 1차 차분**이며, 최적화 전에 상수로 확정된다(결정 무관).
- `vfa/coef.py: coefs()` — 종류별 평균 쪽 shrink 후 [0, coef_hi × max(V,0)]로 절단(가드).
- `match/milp_build.py: build()` — 목적함수 prof + sum_kind c_kind · pv,
  pv[k] = PWL(잉여 s[k])를 `addGenConstrPWL`로 근사. 미래 가치는 **참여자별 완전 분리형(additive)**.
- 관측된 문제: 이 분해에서 보완성(초모듈성)이 소실 → 변환 오차가 시뮬레이션 기준치 표준오차보다 크고
  공급자 유지 가치가 모든 모델에서 일관되게 과소평가.
- 검토 중인 대안: GNN 노드 임베딩 h_i 사전 계산 → z = sum_i p_i h_i (p_i = MILP 내 PWL 재참여확률,
  z는 결정변수의 선형식) → 미래 가치 = g(z), 작은 ReLU 헤드 g만 big-M으로 정확 임베딩.

---

## 1. 증거 표

| # | 제목 | 저자·연도 | 게재지·등급 | 임베딩 방식 | 최적화 모형 | 반복 해결/ADP | 보고된 규모·해 시간 | 우리 방식과의 차이 (한 문장) | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Approximate Dynamic Programming with Neural Networks in Linear Discrete Action Spaces | van Heeswijk, La Poutré, 2019 | arXiv 프리프린트 (저널 게재 표기 없음 → 미확인) | ReLU를 **이진변수 + big-M**으로 정확 임베딩 (Bunel et al. 2018 방식이라 명시) | 정수변수 포함 MILP(적재/하역, 방문 정점) | **예.** ADP 사후상태 VFA로 매 시점 MILP 재해결 | 정점 5개(최대 차수 3), 누적 작업 최대 45, 용량 20. 반복당 0.16초(은닉 1×20), 0.39초(3×20); "계산 예산의 약 99%가 LP 해결" | 비분리형 NN VFA를 MILP에 **직접** 넣는다(우리처럼 개체별 1차 차분으로 분해하지 않음). 단 상태가 저차원 집계이고 참여자 집합이 내생 전이하지 않음 | https://arxiv.org/abs/1902.09855 (전문 https://ar5iv.labs.arxiv.org/abs/1902.09855) | 전문(ar5iv) |
| 2 | Deep Reinforcement Learning in Linear Discrete Action Spaces | van Heeswijk, La Poutré, 2020 | OpenAlex에 venue 미기재 → **미확인** | 미확인 | 미확인 | 미확인 | 미확인 | 미확인 (#1의 학회 버전으로 보이나 확인하지 못함) | https://openalex.org/W3114411349 | 서지만 |
| 3 | The Stochastic Dynamic Postdisaster Inventory Allocation Problem with Trucks and UAVs | van Steenbergen, van Heeswijk, Mes, 2025 | **Transportation Science** 59(2):360–390 (SSCI Q1) | NN-VFA: 은닉 2층 ReLU 신경망을 **big-M 제약으로 단계별 MIP에 직접 임베딩**(Delarue·Anderson·Tjandraatmadja 2020 인용), Gurobi 10.0.2 | 매 기간 MIP | **예.** ADP, 사후상태 특징 | 구역 1–6 및 네팔 13구역. NN-VFA 0.03–0.20초/에폭, DL-VFA 0.002–0.01초/에폭, 완전 재최적화 15분 상한. 최고 비교군 대비 6–8% 개선 | **분리형(DL-VFA)과 비분리형(NN-VFA)을 같은 문제에서 직접 비교한 선례.** 참여자 집합이 결정에 따라 바뀌지 않고, 분해가 "구역별"이며 노드 삭제 차분이 아니다 | https://doi.org/10.1287/trsc.2023.0438 (프리프린트 https://arxiv.org/abs/2312.00140, 읽은 판 https://arxiv.org/html/2312.00140) | 전문 HTML 해당 절 |
| 4 | Shaping Decision Models for Stochastic Dynamic Optimization Problems via Reinforcement Learning | Hildebrandt, Bode, Ulmer, Mattfeld, 2026 | Networks (Wiley, SCIE) | 초록에 "벨만 방정식을 MILP로 모형화하고 가치함수를 신경망으로 근사"라고 명시 | MILP | **예** | 미확인 (전문 페이월 403) | 초록 수준에서 우리와 같은 구조(MILP + NN 가치함수). 세부 정식화·규모·실패 양식 확인 못 함 | https://doi.org/10.1002/net.70039 | 초록만 |
| 5 | Neural Approximate Dynamic Programming for On-Demand Ride-Pooling | Shah, Lowalekar, Varakantham, 2020 | AAAI 2020 (AI 주요 학회) | 학습된 사후상태 가치를 **차량별 개별 가치의 합으로 분해**해 매칭 ILP의 계수로 넣는다 | 매칭 ILP | **예** | 미확인(실도시 데이터, 해 시간 미확인) | **우리와 구조가 가장 같은 선례.** 분해 근거를 "한 에폭 안에서 다른 차량의 행동이 이 차량의 장기 가치를 크게 바꾸지 않는다"로 명시. 이탈이 결정 의존이 아니고 분해 오차를 측정·보고하지 않음 | https://arxiv.org/abs/1911.08842 (전문 https://ar5iv.labs.arxiv.org/html/1911.08842) | 전문의 분해 절 |
| 6 | Reinforcement Learning with Combinatorial Actions: An Application to Vehicle Routing | Delarue, Anderson, Tjandraatmadja, 2020 | **NeurIPS 2020** | ReLU를 이진 지시변수 + big-M + Anderson 외(2020)의 **강화(lifted) 제약**으로 임베딩. big-M은 LP 완화 위에서 사전활성값을 최대·최소화해 조임 | 행동 선택 MIP | **예** (정책 반복) | 은닉 1층 **16 뉴런**. n=21: Gurobi 약 0.4초/MIP, SCIP 약 3초; n=51: Gurobi 약 39초, SCIP 약 235초. 정책 반복 1회(표본경로 250) n=21에서 Gurobi 약 15분(순차), 병렬화 시 약 30초. 뉴런 4개만으로도 선형 모델 대비 유의한 개선 | 가치함수가 **비분리형**(잔여 도시 집합 전체를 입력). 미래 가치 하한으로 목적함수를 max(V_hat, LB1, ...)로 감싸는 가드를 둔다. 참여자 이탈·잉여배분은 없음 | https://proceedings.neurips.cc/paper/2020/file/06a9d51e04213572ef0720dd27a84792-Paper.pdf (읽은 판 https://ar5iv.labs.arxiv.org/html/2010.12001) | 전문(ar5iv) |
| 7 | Mixed-Integer Optimization with Constraint Learning | Maragno, Wiberg, Bertsimas, Birbil, den Hertog, Fajemisin, 2025 | **Operations Research** 73(2):1011–1028 (SSCI Q1) | 학습 모델(선형·트리·앙상블·NN)을 제약·목적에 임베딩(OptiCL) + 신뢰영역(관측 데이터 볼록껍질) | MILP | 아니오(단발) | 미확인 | 정적 단발 최적화. ADP 반복 구조·사후상태 개념 없음 | https://doi.org/10.1287/opre.2021.0707 (프리프린트 https://arxiv.org/abs/2111.04469) | 초록·서지 + 공식 도구 설명 |
| 8 | Optimization with constraint learning: A framework and survey | Fajemisin, Maragno, den Hertog, 2023 | **EJOR** (SSCI Q1) | OCL 5단계 프레임워크와 서베이 | 일반 | 아니오 | 해당 없음 | 동적/ADP 관점이 프레임워크에 없다 | https://doi.org/10.1016/j.ejor.2023.04.041 | 초록 |
| 9 | Strong mixed-integer programming formulations for trained neural networks | Anderson, Huchette, Ma, Tjandraatmadja, Vielma, 2020 | **Mathematical Programming** (SCIE Q1) | ReLU의 이상적(ideal) MIP 정식화 및 강한 완화 | MILP | 아니오 | 미확인(전문 미확인) | 정식화 이론. #6·#3이 이 정식화를 ADP에 가져다 쓴다 | https://doi.org/10.1007/s10107-020-01474-5 | 서지만 |
| 10 | JANOS: An Integrated Predictive and Prescriptive Modeling Framework | Bergman, Huang, Brooks, Lodi, Raghunathan, 2022 | **INFORMS Journal on Computing** (SCIE Q1) | 학습 모델을 MILP에 임베딩하는 모델링 프레임워크 | MILP | 아니오 | 미확인 | 도구. 동적 구조 없음 | https://doi.org/10.1287/ijoc.2020.1023 | 서지만 |
| 11 | OMLT: Optimization & Machine Learning Toolkit | Ceccon, Jalving, Haddad, Thebelt, Tsay, Laird, Misener, 2022 | **JMLR** 23 | 신경망을 Pyomo 블록으로 변환(big-M / 상보성 / 분할 정식화 선택 가능) | MILP/MINLP | 아니오 | 검색 스니펫 수준: 같은 예제 ReLU 망에서 상보성·big-M·분할 정식화가 각각 248·308·428 제약 | 도구. 동적 구조 없음 | https://www.jmlr.org/papers/volume23/22-0277/22-0277.pdf | 서지·스니펫 |
| 12 | Gurobi Machine Learning (공식 문서) | Gurobi, 접속 2026-09-28 | 공식 문서 | scikit-learn·Keras·PyTorch·ONNX·XGBoost·LightGBM 모델 임베딩. Dense+ReLU만, skip/residual 불가 | MILP | 아니오 | **모델 크기 한계나 정식화 선택 지침은 문서에 없음** | 도구. 우리 대안 설계(작은 ReLU 헤드)에 바로 쓸 수 있음 | https://gurobi-machinelearning.readthedocs.io/en/stable/user/supported.html | 전문(해당 페이지) |
| 13 | When Deep Learning Meets Polyhedral Theory: A Survey | Huchette, Muñoz, Serra, Tsay (arXiv 2023; OpenAlex는 INFORMS Journal on Computing 2026으로 표기) | 서베이 (게재지 표기 상충 → **미확인**) | big-M의 약한 LP 완화, 이상적·확장 정식화, 경계 조임 | MILP | 아니오 | 전문 grep에서 "구체적 뉴런 수 상한" 수치를 찾지 못함 | 배경 서베이 | https://arxiv.org/abs/2305.00241 | 전문 부분 grep |
| 14 | Constrained optimization of objective functions determined from random forests | Biggs, Hariss, Perakis, 2022 | **POM** (SSCI Q1) | 랜덤포레스트 목적함수를 MILP로, Benders 컷 | MILP | 아니오 | 미확인 | 트리 앙상블, 정적 | https://doi.org/10.1111/poms.13877 | 초록 |
| 15 | Constraint Learning to Define Trust Regions in Optimization over Pre-Trained Predictive Models | Shi, Emadikhiav, Lozano, Bergman, 2024 | **INFORMS Journal on Computing** 36(6):1382–1399 (SCIE Q1) | 신뢰영역을 제약학습(isolation forest 등)으로 구성 | MILP | 아니오 | 미확인(전문 페이월 403) | **B6 직답 근거.** 신뢰영역 없으면 "최적화해에서의 예측모델 평가를 신뢰할 수 없다"고 명시 | https://doi.org/10.1287/ijoc.2022.0312 (프리프린트 https://arxiv.org/abs/2201.04429) | 초록 |
| 16 | The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis | Smith, Winkler, 2006 | **Management Science** 52(3):311–322 (SSCI Q1) | 해당 없음(개념) | 해당 없음 | 아니오 | 해당 없음 | 추정치가 불편(unbiased)이어도 **최대화 선택 때문에** 선택된 대안의 실제 가치가 추정치보다 작다 — 우리 목적함수가 c_i 오차를 착취할 위험의 원전 | https://doi.org/10.1287/mnsc.1050.0451 | 초록·이차 요약 |
| 17 | Learning Algorithms for Separable Approximations of Discrete Stochastic Optimization Problems | Powell, Ruszczyński, Topaloglu, 2004 | **Mathematics of Operations Research** 29(4):814–836 (SCIE Q1) | 분리 가능 구간선형 오목 VFA를 표본 기울기로 학습(SPAR) | LP/IP(정수해 자연 발생) | **예** | 2단계 확률계획 실험 | **비분리형 문제에 분리형 근사를 쓰는 오차를 명시적으로 다루고 상한(식 56)을 준다.** "두 가지 오차원이 있고 주된 것은 분리형 근사의 사용" | https://doi.org/10.1287/moor.1040.0107 (PDF https://people.orie.cornell.edu/huseyin/publications/spar.pdf) | 전문 부분 추출 |
| 18 | Dynamic-Programming Approximations for Stochastic, Time-Staged Integer Multicommodity-Flow Problems | Topaloglu, Powell, 2006 | **INFORMS Journal on Computing** 18(1):31–42 (SCIE Q1) | 선형 + 구간선형 분리형 VFA | 정수 다품목 최소비용 흐름(매 기간) | **예** | 차량 수·입지 수를 바꾼 문제군 | "대체(substitution)가 있으면 **정확한 가치함수는 비분리형**"이라고 명시하면서도, 분리형 근사(P·PL)가 모든 대체 패턴에서 고품질 해를 준다고 보고 — **우리 갭 주장에 대한 가장 강한 반박 근거** | https://doi.org/10.1287/ijoc.1040.0079 (PDF https://people.orie.cornell.edu/huseyin/publications/imcf.pdf) | 전문 부분 추출 |
| 19 | Approximate Dynamic Programming for Large-Scale Resource Allocation Problems (INFORMS TutORials, 2005) | Powell, Topaloglu | 튜토리얼 챕터 (등급 해당 없음) | 분리형 VFA: 선형 / 구간선형 오목 | LP/IP | **예** | 해당 없음 | "자원 종류가 여럿일 때 좋은 VFA를 어떻게 만들지 분명하지 않다. **단순한 분리형 VFA는 잘 작동하지 않는다**"는 명시적 진술 — 우리 음의 결과와 같은 방향의 기존 지적 | https://people.orie.cornell.edu/huseyin/publications/tutorial_powell_topaloglu_2.pdf | 전문 부분 추출 |
| 20 | Fast Semidifferential-based Submodular Function Optimization | Iyer, Jegelka, Bilmes, 2013 | **ICML 2013** (ML 주요 학회) | 해당 없음(집합함수 이론) | 선형(모듈러) 대리목적 | 반복 | 해당 없음 | **B4의 형식 어휘 원전.** 이탈 한계값 f(j | V\{j}) = f(V) − f(V\{j})로 만드는 supergradient가 모듈러(선형) 상·하한을 정의하고, 그 한계는 **현재 집합에서만 tight** | http://proceedings.mlr.press/v28/iyer13.pdf | 전문 부분 추출 |
| 21 | Optimizing over trained GNNs via symmetry breaking | Zhang, Campos, Feldmann, Walz, Sandfort, Mathea, Tsay, Misener, 2023 | **NeurIPS 2023** | 학습된 GNN을 MIP로 임베딩(그래프가 결정변수인 경우 두 가지 정식화) | MILP | 아니오 | 초록에 규모·시간 없음 | **"고전적 GNN 구조에서 그래프가 고정되면 GNN에 대한 최적화는 밀집 신경망에 대한 최적화와 동등하다"** — 우리처럼 그래프가 결정 무관이면 GNN 특유의 정식화가 필요 없다는 직접 근거 | https://arxiv.org/abs/2305.09420 | 초록 + 인용 문장 |
| 22 | Mixed-Integer Optimisation of Graph Neural Networks for Computer-Aided Molecular Design | McDonald, Tsay, Schweidtmann, Yorke-Smith, 2023(프리프린트)/2024(Computers & Chemical Engineering) | Computers & Chemical Engineering (SCIE) | ReLU GCN용 이중선형 정식화, ReLU GraphSAGE용 MILP 정식화 | MILP / 이중선형 | 아니오 | 초록에 규모·시간 없음 | 분자 설계. GCN이 이중선형이 되는 것(초록 명시)은 그래프 자체가 결정 대상임을 시사 — 우리와 반대 상황 | https://arxiv.org/abs/2312.01228 (저널판 https://www.sciencedirect.com/science/article/pii/S0098135424000784) | 초록 |
| 23 | Graph4BiLO: Graph Neural Network Approximation for Bilevel Mixed-Integer Linear Optimization | Elrefaei, Hua, Kim, Tran, Borrero, 2026 | arXiv 프리프린트 (**미게재**) | GNN 대리모델을 big-M ReLU로 정확 임베딩, **읽기(readout)는 선형** phi_hat = w^T z + b, 풀링은 노드 종류별 평균 후 연결. **풀링이 결정변수에 의존하지 않는다** | 단일수준 MILP | 아니오(이단계 문제) | GNN 2층, 은닉 32. n=100에서 노드 301개, 임베딩된 ReLU 활성 약 28,896개. 해 시간 29.5초(n=20), 1049초(n=40), **n>=60은 3600초 타임아웃** | **GNN 전체를 임베딩하면 규모가 폭발한다는 직접 증거.** 우리 대안(작은 헤드만 임베딩)과 반대 설계. 풀링 가중치가 결정 의존이 아니다 | https://arxiv.org/abs/2608.30103 (읽은 판 https://arxiv.org/html/2608.30103v1) | 전문 HTML |
| 24 | Learning Optimal Dynamic Matching via Graph Neural Networks | Okada, Noda, Komiyama, Matsushita, 2026 | arXiv 프리프린트 (**미게재**) | GNN으로 **사후결정 잔여 그래프**의 연속가치를 근사, TD 학습 | **최적화에 임베딩하지 않음** — forward-greedy 매칭 휴리스틱 | 예(연속시간) | 초록에 규모·시간 없음 | 상태 표현(사후결정 그래프 + GNN)이 가장 비슷하나 (a) 이탈이 **외생**, (b) 조합 최적화에 GNN을 넣지 않고 그리디로 대체, (c) 금전 이전·잉여배분 결정 없음 | https://arxiv.org/abs/2607.28925 | 초록 |
| 25 | Value-Decomposition Networks For Cooperative Multi-Agent Learning | Sunehag 외 10인, 2017 | arXiv 프리프린트(이후 학회 게재 여부 **본 조사 미확인**) | 팀 가치함수를 에이전트별 가치함수로 분해 | 해당 없음 | 예(RL) | 해당 없음 | 초록은 "팀 가치함수를 에이전트별 가치함수로 분해"라고만 말한다. **가산성(additivity) 가정은 초록에 없어 본문 확인 필요** | https://arxiv.org/abs/1706.05296 | 초록 |
| 26 | Counterfactual Multi-Agent Policy Gradients (COMA) | Foerster, Farquhar, Afouras, Nardelli, Whiteson, 2018 | AAAI 2018 | 해당 없음 | 해당 없음 | 예(RL) | 해당 없음 | 개체 제거/대체 차분(difference rewards)을 학습된 비평자로 계산하는 계열. 우리 노드 삭제 차분의 RL 쪽 대응 용어 | https://arxiv.org/abs/1705.08926 | 서지·이차 요약 |
| 27 | Conformal Mixed-Integer Constraint Learning | (저자 미확인), 2025 | arXiv 프리프린트 (**미확인**) | 미확인 | 미확인 | 미확인 | 미확인 | 읽지 않았으므로 내용을 기술하지 않는다 | https://arxiv.org/pdf/2506.03531 | 서지만 |
| 28 | PySCIPOpt-ML: Embedding Trained Machine Learning Models into Mixed-Integer Programs | (저자 미확인), 2023 | arXiv 프리프린트 (**미확인**) | 미확인 | MILP | 아니오 | 미확인 | 읽지 않았으므로 내용을 기술하지 않는다 | https://arxiv.org/abs/2312.08074 | 서지만 |

주의: 20번 항목의 "f(j | V\{j})" 표기는 표 구분자와 겹치므로, 정확한 표기는 §2 B4를 보라.

---

## 2. B1–B6 직답

### B1. 신경망을 최적화에 정확히 임베딩하는 현재 표준과 보고된 계산 한계

**표준은 확립되어 있다.** ReLU 망은 은닉 뉴런마다 이진 지시변수 하나와 big-M 제약으로 정확히 표현되며[1][6],
big-M의 약한 LP 완화를 개선하는 이상적(ideal)·확장 정식화가 Anderson 외(Mathematical Programming 2020)에 있다[9][13].
도구 계층은 JANOS(IJOC 2022)[10], OptiCL/Maragno 외(Operations Research 2025)[7], OMLT(JMLR 2022)[11],
gurobi-machinelearning[12]이며 프레임워크·서베이는 Fajemisin 외(EJOR 2023)[8]이다.
gurobi-machinelearning 공식 문서는 Dense+ReLU만 지원하고 skip/residual 연결을 허용하지 않으며,
**모델 크기 한계에 대한 지침은 문서에 없다**[12].

**보고된 계산 한계 (구체 수치가 있는 것만)**
- Delarue 외(NeurIPS 2020): 은닉 1층 **16 뉴런**, 입력 차원 n=21에서 Gurobi 약 0.4초/MIP, SCIP 약 3초.
  n=51에서 Gurobi 약 39초, SCIP 약 235초. 뉴런 수를 늘릴 때 해 시간이 급격히 증가[6].
- van Heeswijk & La Poutré: 은닉 1×20과 3×20에서 ADP 반복당 0.16초·0.39초, 계산의 약 99%가 LP/MILP 해결[1].
- van Steenbergen 외(Transportation Science 2025): 은닉 2층 NN을 MIP에 넣어 **0.03–0.20초/의사결정 에폭**[3].
- Graph4BiLO(2026 프리프린트): GNN 2층·은닉 32를 통째로 임베딩하면 n=100에서 ReLU 활성 약 28,896개,
  n=40에서 1049초, **n>=60에서 1시간 타임아웃**[23].
- "When Deep Learning Meets Polyhedral Theory" 서베이에서는 **구체적 뉴런 수 상한 수치를 찾지 못했다**
  (확인한 것은 big-M 완화가 약해 분기 트리가 커진다는 정성적 서술)[13]. 이 항목은 미해결로 남긴다.

### B2. 그 임베딩을 주기별로 반복 해결하는 ADP/rollout 구조에 쓴 연구가 있는가 — **있다 (4편 확인)**

1. van Heeswijk & La Poutré (2019, arXiv): 신경망 VFA를 big-M으로 MILP에 임베딩하고 ADP로 반복 해결[1].
2. Delarue, Anderson, Tjandraatmadja (NeurIPS 2020): 가치함수를 MIP에 임베딩해 행동을 선택하고 정책 반복[6].
3. van Steenbergen, van Heeswijk, Mes (Transportation Science 2025): NN-VFA를 매 기간 MIP에 임베딩, ADP[3].
4. Hildebrandt, Bode, Ulmer, Mattfeld (Networks 2026): 초록에 "벨만 방정식을 MILP로 모형화하고 가치함수를 신경망으로 근사"[4].

따라서 **"학습된 V를 사후상태 가치로 두고 MILP가 직접 최대화한다"는 것 자체는 우리 기여가 아니다.**
이것은 갭 주장을 좁히는 데 결정적이다.

### B3. 【핵심】 비분리형(nonseparable) VFA를 정수계획에 넣은 연구가 있는가 — **있다. 다루는 방식은 ReLU big-M 정확 임베딩이다**

- ADP 표준 근사 구조는 분리 가능 오목·구간선형 VFA(Powell–Ruszczyński–Topaloglu MOR 2004[17],
  Topaloglu–Powell IJOC 2006[18], Powell–Topaloglu 튜토리얼[19])와 선형 기저함수[19]이다.
- **비분리형을 IP에 넣은 선례는 존재한다**: [1][3][6]. 방식은 모두 동일하다 —
  ReLU를 이진변수 + big-M(및 강화 제약)으로 바꿔 **MILP로 정확히 표현**한다.
- 분리형 근사의 오차를 정면으로 다룬 고전은 SPAR[17]이다. 비분리형 문제에 분리형 근사를 쓰면
  최적성을 보장하지 못하며, 저자들은 오차 상한(논문 식 56)을 제시하고
  "두 가지 오차원이 있는데 **주된 것은 분리형 근사의 사용**"이라고 적는다[17].
  단 그 논증은 연속·오목 구조에 기반하며, 우리처럼 **이산 참여자 집합에 대한 집합함수** 상황은 다루지 않는다(추론).
- Powell–Topaloglu 튜토리얼은 더 직접적이다: "자원 종류가 여럿일 때 좋은 VFA를 어떻게 구성할지 분명하지 않다.
  연구 결과는 단순한 분리형 VFA가 **잘 작동하지 않는다**는 것을 보여준다"[19].
  우리 설정의 "여러 품목 × 여러 공급자"는 이 "다중 자원 종류"와 구조적으로 대응한다(추론).
- **반대 방향 증거도 명확히 있다**: Topaloglu–Powell IJOC 2006은 "대체가 있으면 정확한 가치함수는 비분리형"이라고
  인정하면서도, 분리형 구간선형 근사가 **모든 대체 패턴에서 고품질 해**를 준다고 보고한다[18].
  즉 "비분리성이 있다 → 분리형 근사가 나쁘다"는 자동으로 성립하지 않는다.

### B4. 【핵심】 노드/자원 삭제 1차 차분 근사의 기존 용어와 편향 지적

**기존 용어 — 세 문헌군에 이미 있다.**
1. **ADP/확률계획**: 좌표 방향 e_i의 표본 기울기(sample gradients)로 만드는 **분리형 구간선형 근사**[17].
   기울기·한계가치(marginal value)로 분리형 근사의 기울기를 추정하는 것이 표준이다[17][19].
2. **집합함수 이론**: 우리 c_i = V(S) − V(S\{i})는 Iyer–Jegelka–Bilmes의
   **discrete semigradient(supergradient)** 성분 g_hat(j) = f(j | V\{j})와 정확히 같다.
   이것이 정의하는 **모듈러(선형) 상·하한**은 m_{g_Y}(X) = f(Y) + g_Y(X) − g_Y(Y)이고
   논문은 "이 두 한계는 **현재 해에서 tight**하다(m(Y) = f(Y))"고 적는다[20].
   즉 참여자 일부가 떠난 집합에서는 근사가 어긋나는 것이 정의상 예견된다.
3. **다중 에이전트 RL**: 팀 가치를 에이전트별 가치로 분해하는 value decomposition[25]과
   개체 제거/대체 차분(difference rewards)을 학습된 비평자로 계산하는 COMA[26].
   (단 VDN 초록에는 가산성 가정이 명시되어 있지 않다 — 본문 확인 필요[25].)

**편향이 이미 지적되었는가 — 부분적으로 그렇다.**
- 분리형 근사가 상호작용이 있을 때 오차를 낸다는 지적은 ADP에 **이미 있다**[17][19].
  특히 [19]의 "다중 자원 종류에서 단순 분리형 VFA는 잘 작동하지 않는다"가 가장 가깝다.
- 모듈러 근사가 현재 집합에서만 tight하다는 형식적 진술도 **이미 있다**[20].
- NeurADP는 개체별 분해를 하면서 그 가정을 **명시적으로 서술**한다: 한 에폭 안에서 다른 차량의 행동이
  이 차량의 장기 가치를 크게 바꾸지 않으므로 교차 상호작용 항을 제거한다[5].
  그러나 (확인한 범위에서) **그 분해 오차를 직접 측정해 보고하지는 않는다**.
- **찾지 못한 것**: "학습된 신경망 V에 대해 노드 삭제 1차 차분으로 개체별 계수를 만들고,
  그 변환 오차를 시뮬레이션 기준치와 대조해 정량화하고, 초모듈성 때문에 특정 종류의 참여자 계수가
  체계적으로 과소평가된다고 보고한" 연구. 아래 §4의 인용 그래프 검사에서도 이 교집합이 비어 있다.

### B5. GNN 임베딩을 최적화 모형에 결합한 사례 / 결정 무관 그래프 + 결정 의존 생존확률의 선형 읽기

- **GNN을 MILP에 임베딩한 사례는 있다**: 분자 설계(ReLU GCN 이중선형, GraphSAGE MILP)[22],
  대칭 파괴로 학습된 GNN에 대한 최적화[21], 이단계 최적화의 GNN 대리모델[23].
- **결정적 사실**: [21]은 "고전적 GNN 구조에서 **그래프가 고정되면** GNN에 대한 최적화는
  밀집 신경망에 대한 최적화와 **동등**하다"고 명시한다. 그래서 그들은 각 간선이 결정변수인 경우만 연구한다[21].
  우리 설정은 그래프가 결정 무관이므로, 이 진술에 따르면 GNN 전용 정식화가 필요하지 않다.
- **선형 읽기(readout) 선례는 있다**: Graph4BiLO는 노드 표현을 먼저 집계(종류별 평균 풀링)한 뒤
  선형 읽기 phi_hat = w^T z + b를 쓰며, 풀링은 결정변수에 의존하지 않는다[23].
  즉 "선형 읽기" 자체는 선례가 있으나, **풀링 가중치가 결정변수(생존확률)인 구조는 찾지 못했다.**
- 가장 가까운 것은 Okada 외(2026): GNN으로 사후결정 잔여 그래프의 가치를 근사한다.
  그러나 이탈이 **외생**이고, 조합 최적화에 GNN을 넣지 않고 **forward-greedy 휴리스틱**으로 대체한다[24].
- 따라서 z = sum_i p_i h_i (p_i가 MILP 내 결정 의존 확률)로 기대 그래프 표현을 만들고 작은 헤드만 임베딩하는 구조는
  **확인한 범위에서 선례를 찾지 못했다**. §5에 확인한 쿼리를 남긴다.

### B6. 학습된 목적함수를 최적화에 넣을 때 보고되는 실패 양식

1. **외삽·신뢰 불가**: 신뢰영역 제약이 없으면 "최적화해에서의 예측모델 평가를 신뢰할 수 없고 해의 실용성이
   비합리적일 수 있다"[15]. OptiCL은 관측 데이터의 볼록껍질로 신뢰영역을 잡는다[7].
   Shi 외(IJOC 2024)는 isolation forest로 학습한 신뢰영역이 기존 방식보다 낫다고 보고한다[15].
2. **최적화가 추정 오차를 착취**: Smith–Winkler의 optimizer's curse — 추정치가 불편이어도 최대화 선택 때문에
   선택된 대안의 실제 가치가 추정치보다 작다[16].
3. **가드 장치의 실제 사례**: Delarue 외는 목적함수의 학습된 가치 V_hat을 알려진 하한으로 감싸
   max(V_hat, LB1, ..., LBP)로 쓰고, 이것이 성능을 개선한다고 보고한다.
   반면 **신뢰영역이나 불확실성 정량화는 다루지 않는다**[6].
4. RL 쪽의 함수근사 과대추정 편향(overestimation bias) 계열 논의가 있으나 본 조사에서 전문을 읽지 않았으므로
   근거로 쓰지 않는다(§5 미해결).

우리 구현의 가드(`coefs()`의 shrink + [0, coef_hi·max(V,0)] 절단)는 [6]의 하한 감싸기와 [15]의 신뢰영역 중
**전자에 가까운 사후 절단**이며, 입력 공간 신뢰영역은 아니다(코드 근거 + 추론).

---

## 3. 계산 가능성 근거: 우리 규모는 들어가는가

**우리 규모** (품목 6, 공급자 6, 주문자 약 30, 은닉층 폭 8–16, 헤드 1–2층):
- 대안 설계에서 MILP에 들어가는 것은 **헤드 g만**이다. 은닉 폭 16, 1–2층이면 ReLU 이진변수 16–32개.
- 비교 기준: Delarue 외는 은닉 1층 16 뉴런에서 Gurobi 0.4초(입력 21)·39초(입력 51)를 보고한다[6].
  우리 헤드의 입력은 GNN 임베딩 차원 z(설계 선택, 예: 16–32)이며 Delarue의 입력 규모와 같은 자릿수이다.
- van Steenbergen 외는 은닉 2층 NN을 매 기간 MIP에 넣어 0.03–0.20초/에폭을 얻었다[3].
- van Heeswijk는 은닉 3×20에서도 ADP 반복당 0.39초를 보고한다[1].

**판단**: 헤드만 임베딩하는 설계는 보고된 실용 범위 안에 **명확히 들어간다**.
근거는 [1][3][6]의 은닉 16–60 뉴런 규모에서 초 단위 해 시간이다.
반대로 **GNN 전체를 임베딩하는 설계는 위험하다**: Graph4BiLO는 GNN 2층·은닉 32에서
노드 301개일 때 ReLU 활성 약 28,896개, n>=60에서 1시간 타임아웃을 보고한다[23].
우리 노드 수(주문자 30 + 공급자 6 = 36)에 은닉 32를 곱하면 층당 활성이 1,000개를 넘어
Graph4BiLO의 n=40(1049초) 근방으로 들어간다(추론 — 우리가 직접 실험한 값이 아니다).
또한 [21]의 "그래프 고정 → 밀집망과 동등"에 따르면 우리 설정에서 GNN 전체 임베딩은 이득 없이 비용만 늘린다.

**주의 1**: 위 비교는 ReLU 이진변수 수만 맞춘 것이다. 우리 MILP에는 배정 정수변수
x[b,j,i](주문자 30 × 공급자 6 × 품목 6 약 1,080개)와 참여자별 PWL 제약이 이미 있으므로
결합 후 해 시간은 위 문헌 수치보다 커진다. **결합 규모의 해 시간을 직접 보고한 문헌은 찾지 못했다.**
**주의 2**: big-M 값의 품질이 해 시간을 크게 좌우하며 [6]은 LP 완화 위의 경계 조임으로 해결한다.
우리 z = sum_i p_i h_i는 p_i in [0,1], h_i가 상수이므로 z의 상·하한을 **해석적으로 정확히** 줄 수 있다
(추론, 아직 구현 안 됨).

---

## 4. 인용 그래프에서 관찰한 것 (갭 신호)

OpenAlex `filter=cites:A+cites:B`로 두 군집의 **공통 인용자** 수를 세었다 (2026-09-28 실행).

| 씨앗 A (제약학습·NN 정식화) | 씨앗 B (ADP 근사구조) | 공통 인용자 수 |
|---|---|---|
| Anderson 외 2020 (W2901816197) | NeurADP (W3033180544) | **0** |
| Maragno 외 OR (W3211516017) | NeurADP (W3033180544) | **0** |
| JANOS (W3199600733) | NeurADP (W3033180544) | **0** |
| Maragno 외 OR | Powell ADP 책 (W100327610) | **0** |
| Maragno 외 OR | Powell ADP 책 (W2024780670) | **0** |
| Anderson 외 2020 | Powell ADP 책 (W100327610) | **0** |
| Anderson 외 2020 | Topaloglu–Powell IJOC 2006 (W2149052950) | **0** |
| Maragno 외 OR | Powell–Ruszczyński–Topaloglu 2004 (W2171027013) | **0** |
| Iyer 외 2013 (W1510950165) | NeurADP / Maragno / Anderson / Powell 책 | **각 0** |

관찰:
1. **제약학습(OCL) 군집과 ADP 분리형 근사 군집은 OpenAlex 상에서 서로를 함께 인용하지 않는다.**
   "학습 모델을 최적화에 넣는 법"과 "ADP에서 가치함수를 최적화에 넣는 법"은 사실상 별개 문헌으로 자라났다.
2. 두 군집을 실제로 잇는 경로는 Anderson 외의 **정식화**를 ADP 응용이 가져다 쓰는 것뿐이다:
   Anderson 외 2020 → Delarue 외 2020 → van Steenbergen 외 2025(TS)[6][3].
   이 경로에 OCL 프레임워크(OptiCL/JANOS/OMLT)는 등장하지 않는다.
   Anderson 인용자를 "dynamic programming value function"으로 좁힌 결과는 32건이며,
   ADP 응용은 [3]과 Hildebrandt 외[4]를 포함한 소수다.
3. Maragno 외 인용자 60건을 "dynamic / multiperiod / value function"으로 좁히면 **0건**이다.
   OCL 문헌은 아직 다기간 동적 의사결정으로 확장되지 않았다.
4. **집합함수 semigradient 문헌[20]은 어느 군집과도 공통 인용자가 없다.**
   노드 삭제 차분의 편향을 형식적으로 해석하는 어휘가 ADP/VFA 문헌에 아직 수입되지 않았다는 신호다.

**갭 신호 요약**: 주장할 수 있는 빈 칸은 "비분리형 VFA를 IP에 넣는 것"(이미 있음: [1][3][6])이 아니라
(a) **분리형 변환 오차를 시뮬레이션 기준치로 정량화하고 그 편향 방향을 구조(초모듈성)로 설명하는 것**,
(b) **참여자 집합이 결정 의존적으로 전이하는 상태(= 참여자별 생존확률이 결정변수)에서
     그래프 표현의 기대값을 결정변수의 선형식으로 만들고 헤드만 임베딩하는 구조**,
(c) OCL 문헌의 신뢰영역·가드 장치를 ADP 반복 해결 루프로 가져오는 것 — 세 가지다.

---

## 5. 미해결·확인 불가 (페이월 목록과 확인할 구체적 질문)

| 항목 | 상태 | 확인할 구체적 질문 |
|---|---|---|
| Hildebrandt 외, Networks 2026[4] | **페이월 403**(Wiley) | 세 번째 방법의 MILP 정식화는 big-M인가 이상적 정식화인가? 신경망 크기? 해 시간? 가치함수 오차를 최적화가 착취하는 현상을 보고하는가? **B2의 최근접 선례이므로 최우선 확인 대상** |
| Shi 외, IJOC 2024[15] | **페이월 403**, 프리프린트 arXiv:2201.04429 존재(미독) | 신뢰영역 세 방식의 정의와 MILP 크기 증가량. 우리 c_i 가드를 신뢰영역으로 바꿀 수 있는가 |
| Anderson 외, Math Prog 2020[9] | 전문 미확인 | 실험에서 쓴 망 크기와 big-M 대비 이상적 정식화의 해 시간 배수. 우리 헤드 크기에서 이상적 정식화를 쓸 가치가 있는가 |
| Maragno 외, OR 2025[7] | 초록만 | 신뢰영역 정식화의 MILP 비용. 임베딩한 NN의 최대 크기와 해 시간 |
| NeurADP[5] | 분해 절만 읽음 | 분해 오차를 어디서든 정량화했는가? 차량 수·해 시간? **B4 선행성 판정에 직결** |
| Topaloglu–Powell 2006[18] | 실험 절만 추출 | 분리형 근사가 대체 패턴 전체에서 잘 작동한 이유에 대한 저자들의 설명. 우리 반박 대응 논거 |
| VDN[25] 본문 | 초록만 | 가산성 가정이 본문에 명시되는가, 게재 학회 확정 |
| 폴리헤드럴 서베이[13] | 부분 grep | 게재지 확정(arXiv 2023 vs OpenAlex의 IJOC 2026)과 계산 한계 절의 구체 수치 |
| van Heeswijk & La Poutré[1] 게재 여부 | **미확인** | arXiv 1902.09855의 저널·학회 게재판 존재 여부. OpenAlex W3114411349와 동일 논문인가 |
| Conformal Mixed-Integer Constraint Learning[27], PySCIPOpt-ML[28] | 서지만 | 저자·연도·게재 여부. B1·B6 보강용 |
| RL 과대추정 편향 문헌 | **미확인** | 함수근사 오차 → 과대추정 편향을 정식으로 진술한 원전(TD3/Double Q-learning)을 전문 확인 후 B6에 넣을지 결정 |

**검색했으나 결과가 없었던 것 (그대로 기록)**
- OpenAlex 공통 인용자 검사 9건 전부 **0건** (§4 표).
- Maragno 외 인용자를 `search=dynamic multiperiod value function`으로 좁힌 결과 **0건**.
- 다음 쿼리로 "결정 의존 생존확률로 노드 임베딩을 가중 합해 만든 기대 그래프 표현을 MILP에 넣은" 연구를 찾지 못했다:
  `graph neural network value function approximation variable number of agents dynamic matching embedded integer program platform`;
  `"piecewise linear" retention probability response function embedded MILP objective platform revenue sharing dynamic participant retention value`;
  `graph neural network embedded in mixed integer program surrogate objective optimization over trained GNN`.
  가장 가까운 [24]는 GNN을 최적화에 넣지 않고 그리디로 대체하며 이탈이 외생이다.
- `value function decomposition individual agent marginal contribution linear integer program ignores complementarity supermodular bias`
  검색은 MARL 문헌만 반환했고, **OR/ADP 쪽에서 노드 삭제 차분의 초모듈성 편향을 직접 지적한 논문은 찾지 못했다.**
- `"nonseparable" value function approximation integer program approximate dynamic programming difficulty embedding`
  검색에서 SPAR 계열([17]) 외에 새로운 선례를 찾지 못했다.

---

## 6. 우리 음의 결과의 위치 (B4 답에 따른 판정)

- "분리형/모듈러 근사가 상호작용이 있을 때 부정확하다"는 **이미 지적된 한계**다[17][19][20].
  따라서 우리 관측을 "처음 발견한 현상"으로 쓸 수 없다.
- 새로 주장할 수 있는 범위(확인한 범위에서 선례 없음):
  (a) 노드 삭제 차분의 변환 오차를 **시뮬레이션 기준치와 그 표준오차 대비로 정량화**한 것,
  (b) 그 편향이 **공급자 쪽에서 체계적으로 과소평가 방향**이라는 관측을 묶음 주문의 보완성과 연결한 것,
  (c) 참여자 생존확률이 결정변수인 상태 전이에서 분리형 변환을 **우회하는** 임베딩 구조.
- 반박 가능성: Topaloglu–Powell[18]을 근거로 "비분리성이 있어도 분리형 근사는 잘 작동한다"고 말할 수 있다.
  대응 논거는 [19]의 다중 자원 종류 진술과 우리 실험의 변환 오차 수치다.
  또 [21]의 "그래프 고정 → 밀집망과 동등"을 근거로 "GNN이 필요 없다"고 말할 수 있다 —
  이것은 실험 3의 설계와 직접 충돌하므로 논문에서 먼저 다루어야 한다.

---

## Coverage Status

| 질문 | 상태 | 비고 |
|---|---|---|
| B1 표준·계산 한계 | **done** (일부 미해결) | 표준 확정. 서베이의 구체적 뉴런 상한 수치는 찾지 못함 |
| B2 ADP 반복 해결 임베딩 선례 | **done** | 4편 확인, 그중 3편 전문/부분 전문 |
| B3 비분리형 VFA를 IP에 | **done** | 선례 있음. 방식은 ReLU big-M. 반박 근거([18])도 확보 |
| B4 노드 삭제 차분의 용어·편향 지적 | **done** | 용어 3계열 확보. 편향 지적은 부분적으로 기존, 정량화 사례는 못 찾음 |
| B5 GNN 임베딩·선형 읽기 | **done** | GNN-MILP 선례 있음. 결정 의존 풀링 가중치 선례 못 찾음 |
| B6 실패 양식·가드 | **done** (일부 미해결) | 신뢰영역·optimizer's curse·하한 감싸기 확보. RL 과대추정 편향 원전 미확인 |
| 인용 그래프 교집합 검사 | **done** | 9개 조합 전부 0건 |
| 페이월 항목 | **blocked** | §5 표 |

**직접 확인한 것**: [1][3][5][6][12][17][18][19][20][21][23][24] (전문 또는 해당 절 직접 추출),
[7][8][14][15][22][25] (초록), [4] (초록만), [9][10][11][13][16][26][27][28] (서지 또는 부분).
**직접 확인하지 못한 것**: §5 표.

---

## 7. 출처 목록

1. van Heeswijk, W., La Poutré, H. (2019) *Approximate Dynamic Programming with Neural Networks in Linear Discrete Action Spaces* — https://arxiv.org/abs/1902.09855 (읽은 판 https://ar5iv.labs.arxiv.org/abs/1902.09855)
2. van Heeswijk, W., La Poutré, H. (2020) *Deep Reinforcement Learning in Linear Discrete Action Spaces* — https://openalex.org/W3114411349
3. van Steenbergen, R., van Heeswijk, W., Mes, M. (2025) *The Stochastic Dynamic Postdisaster Inventory Allocation Problem with Trucks and UAVs*, Transportation Science 59(2):360–390 — https://doi.org/10.1287/trsc.2023.0438 · 프리프린트 https://arxiv.org/abs/2312.00140
4. Hildebrandt, F., Bode, A., Ulmer, M., Mattfeld, D. (2026) *Shaping Decision Models for Stochastic Dynamic Optimization Problems via Reinforcement Learning*, Networks — https://doi.org/10.1002/net.70039
5. Shah, S., Lowalekar, M., Varakantham, P. (2020) *Neural Approximate Dynamic Programming for On-Demand Ride-Pooling*, AAAI 2020 — https://arxiv.org/abs/1911.08842
6. Delarue, A., Anderson, R., Tjandraatmadja, C. (2020) *Reinforcement Learning with Combinatorial Actions: An Application to Vehicle Routing*, NeurIPS 2020 — https://proceedings.neurips.cc/paper/2020/file/06a9d51e04213572ef0720dd27a84792-Paper.pdf
7. Maragno, D., Wiberg, H., Bertsimas, D., Birbil, Ş.İ., den Hertog, D., Fajemisin, A. (2025) *Mixed-Integer Optimization with Constraint Learning*, Operations Research 73(2):1011–1028 — https://doi.org/10.1287/opre.2021.0707 · 프리프린트 https://arxiv.org/abs/2111.04469
8. Fajemisin, A., Maragno, D., den Hertog, D. (2023) *Optimization with constraint learning: A framework and survey*, EJOR — https://doi.org/10.1016/j.ejor.2023.04.041
9. Anderson, R., Huchette, J., Ma, W., Tjandraatmadja, C., Vielma, J.P. (2020) *Strong mixed-integer programming formulations for trained neural networks*, Mathematical Programming — https://doi.org/10.1007/s10107-020-01474-5
10. Bergman, D., Huang, T., Brooks, P., Lodi, A., Raghunathan, A. (2022) *JANOS: An Integrated Predictive and Prescriptive Modeling Framework*, INFORMS Journal on Computing — https://doi.org/10.1287/ijoc.2020.1023
11. Ceccon, F., Jalving, J., Haddad, J., Thebelt, A., Tsay, C., Laird, C.D., Misener, R. (2022) *OMLT: Optimization & Machine Learning Toolkit*, JMLR 23 — https://www.jmlr.org/papers/volume23/22-0277/22-0277.pdf
12. Gurobi *Machine Learning — Supported models* (접속 2026-09-28) — https://gurobi-machinelearning.readthedocs.io/en/stable/user/supported.html
13. Huchette, J., Muñoz, G., Serra, T., Tsay, C. *When Deep Learning Meets Polyhedral Theory: A Survey* — https://arxiv.org/abs/2305.00241
14. Biggs, M., Hariss, R., Perakis, G. (2022) *Constrained optimization of objective functions determined from random forests*, Production and Operations Management — https://doi.org/10.1111/poms.13877
15. Shi, C., Emadikhiav, M., Lozano, L., Bergman, D. (2024) *Constraint Learning to Define Trust Regions in Optimization over Pre-Trained Predictive Models*, INFORMS Journal on Computing 36(6):1382–1399 — https://doi.org/10.1287/ijoc.2022.0312 · 프리프린트 https://arxiv.org/abs/2201.04429
16. Smith, J.E., Winkler, R.L. (2006) *The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis*, Management Science 52(3):311–322 — https://doi.org/10.1287/mnsc.1050.0451
17. Powell, W.B., Ruszczyński, A., Topaloglu, H. (2004) *Learning Algorithms for Separable Approximations of Discrete Stochastic Optimization Problems*, Mathematics of Operations Research 29(4):814–836 — https://doi.org/10.1287/moor.1040.0107 · PDF https://people.orie.cornell.edu/huseyin/publications/spar.pdf
18. Topaloglu, H., Powell, W.B. (2006) *Dynamic-Programming Approximations for Stochastic, Time-Staged Integer Multicommodity-Flow Problems*, INFORMS Journal on Computing 18(1):31–42 — https://doi.org/10.1287/ijoc.1040.0079 · PDF https://people.orie.cornell.edu/huseyin/publications/imcf.pdf
19. Powell, W.B., Topaloglu, H. *Approximate Dynamic Programming for Large-Scale Resource Allocation Problems* (INFORMS TutORials, 2005) — https://people.orie.cornell.edu/huseyin/publications/tutorial_powell_topaloglu_2.pdf
20. Iyer, R., Jegelka, S., Bilmes, J. (2013) *Fast Semidifferential-based Submodular Function Optimization*, ICML 2013 — http://proceedings.mlr.press/v28/iyer13.pdf
21. Zhang, S., Campos, J.S., Feldmann, C., Walz, D., Sandfort, F., Mathea, M., Tsay, C., Misener, R. (2023) *Optimizing over trained GNNs via symmetry breaking*, NeurIPS 2023 — https://arxiv.org/abs/2305.09420
22. McDonald, T., Tsay, C., Schweidtmann, A.M., Yorke-Smith, N. (2023/2024) *Mixed-Integer Optimisation of Graph Neural Networks for Computer-Aided Molecular Design* — https://arxiv.org/abs/2312.01228 · 저널판 https://www.sciencedirect.com/science/article/pii/S0098135424000784
23. Elrefaei, J.D., Hua, K., Kim, S., Tran, H.N., Borrero, J.S. (2026) *Graph4BiLO: Graph Neural Network Approximation for Bilevel Mixed-Integer Linear Optimization* — https://arxiv.org/abs/2608.30103
24. Okada, G., Noda, S., Komiyama, J., Matsushita, A. (2026) *Learning Optimal Dynamic Matching via Graph Neural Networks* — https://arxiv.org/abs/2607.28925
25. Sunehag, P. 외 (2017) *Value-Decomposition Networks For Cooperative Multi-Agent Learning* — https://arxiv.org/abs/1706.05296
26. Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., Whiteson, S. (2018) *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018 — https://arxiv.org/abs/1705.08926
27. *Conformal Mixed-Integer Constraint Learning* (2025, 프리프린트) — https://arxiv.org/pdf/2506.03531
28. *PySCIPOpt-ML: Embedding Trained Machine Learning Models into Mixed-Integer Programs* (2023, 프리프린트) — https://arxiv.org/abs/2312.08074
