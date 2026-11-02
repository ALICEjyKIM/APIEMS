# 주제 C 증거 수집: 구조 정보가 요약 지표를 이기는 조건 + Neur2RO 계보

조사일 2026-09-28. 목적은 문헌 요약이 아니라 **리서치 갭 특정**이다.
1차(A·B)에서 답이 나온 것(B1 신경망 MILP 정확 임베딩 표준·계산 한계, B4 노드 삭제 1차 차분의 기존 용어와 편향,
B5 GNN 전체를 MILP에 임베딩한 사례)은 여기서 다시 묻지 않는다.

읽은 기준 파일: `CLAUDE.md`(실험 3·그래프 설계), `ref/literature-review.md`(C1~C7),
`ref/_review/..._research_B.md`(§2 B5, §3, §4), `docs/tex/facts.md`(`exp3_diag.json` 절).

우리 실험 3 v1 실제 수치(`exp3_diag.json`, wd 0.1 검증 R² 최고):
conc0_k1 MLP 0.394 / MLPconc 0.393 / GNN 0.376, conc0_k2 0.307 / 0.298 / 0.241,
conc1_k1 0.154 / 0.160 / **0.0073**, conc1_k2 0.386 / 0.388 / 0.279. 잡음 상한 R² 0.426~0.487.
즉 GNN이 네 칸 모두 최하위, conc1_k1에서 사실상 0. 진단된 원인 후보: 입력 척도 결함(공급자 수량 최대 |z| 10~11),
합 풀링의 크기 의존(주문자 수 4~58).

---

## 1. 증거 표

| # | 제목 | 저자·연도 | 게재지·등급 | 내용 요약 | 우리와의 차이 한 문장 | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|
| 1 | Neur2SP: Neural Two-Stage Stochastic Programming | Dumouchelle, Patel, Khalil, Bodur, 2022 | NeurIPS 2022 (ML 최상위) | 시나리오별 임베딩 Ψ1 → **평균 집계** → Ψ2 → 1단 결정 x와 결합한 Φ_E만 MIP에 임베딩. "only the latent representation is embedded into the approximate MIP" | 집계 가중치가 결정과 무관(시나리오 확률은 상수)이고, 참여자 집합 전이나 다기간 루프가 없다 | https://arxiv.org/abs/2205.12006 | 전문(구조·임베딩 절) |
| 2 | Neur2RO: Neural Two-Stage Robust Optimization | Dumouchelle, Julien, Kurtz, Khalil, 2024 | ICLR 2024 (ML 최상위) | 1단 변수별·시나리오 변수별 임베딩을 공유 파라미터망으로 만들어 집계 후 작은 값망 Φ. 주문제에서는 "the scenario embeddings can be precomputed via a forward pass for each scenario, i.e., no MILP representation is needed", 적대문제에서는 x 임베딩을 사전 계산 | **결정 의존 부분만 임베딩·결정 무관 부분 사전 계산**이라는 우리 대안 설계의 직접 선례이며, 집계 항이 결정변수의 비선형 함수라는 점에서 우리보다 더 일반적이다 | https://arxiv.org/html/2310.04345v2 | 전문(구조·임베딩·논의 절) |
| 3 | Neur2BiLO: Neural Bilevel Optimization | Dumouchelle, Julien, Kurtz, Khalil, 2024 | NeurIPS 2024 (ML 최상위) | "Embedding the set of variable features using a set-based architecture ... summing up the resulting n variable embeddings", "akin to the DeepSets approach". InstanceEmbedding은 사전 계산, 변수별 임베딩은 배정값에 의존. 실험의 상위 결정변수 유형에 **연속(C)** 포함(donor-recipient) | **합 풀링의 항이 연속 결정변수에 의존하는 사례가 이미 존재한다.** 우리 z = Σ p_i h_i는 h_i가 상수여서 오히려 선형인 특수형 | https://arxiv.org/html/2402.02552v2 | 전문(구조·실험 표) |
| 4 | Reinforcement learning with combinatorial actions for coupled restless bandits (SEQUOIA) | Xu, Wilder, Khalil, Tambe, 2025 | ICLR 2025 (ML 최상위) | "we embed the trained Q-network into the MILP formulations of our constraints and use its output as our objective". 2층×32 ReLU MLP, 행동은 **이진 벡터**, 풀링·개체별 분해 없음. 응용에 이분 매칭·용량 제약 포함 | 학습 Q를 MILP에 넣고 매 기간 푸는 ADP 루프는 이미 있으나, 행동이 이진이고 집합 풀링이 없으며 참여자별 계수 분해를 하지 않는다 | https://arxiv.org/html/2503.01919v1 | 전문(구조·MILP·하이퍼파라미터) |
| 5 | A Fair Comparison of Graph Neural Networks for Graph Classification | Errica, Podda, Bacciu, Micheli, 2019/2020 | ICLR 2020 (ML 최상위) | 구조 무관 기준선 = "global sum pooling ... then applies a single-layer MLP"(화학), "a single-layer MLP on top of node features, followed by global sum pooling and another single-layer MLP"(소셜). 10-fold 외부 CV + 내부 holdout 모델선택 + 최종 3회 재학습, 그리드 32~72 구성. "on D&D, PROTEINS and ENZYMES none of the GNNs are able to improve over the baseline" | 기준선이 **우리 시장 요약 지표 MLP와 사실상 동일한 구조(합 풀링 + MLP)** 이고, 9개 중 3개에서 GNN이 이를 못 넘었다 | https://arxiv.org/abs/1912.09893 | 전문(기준선·프로토콜·결과 절) |
| 6 | Graph Neural Networks Use Graphs When They Shouldn’t | Bechler-Speicher, Amos, Gilad-Bachrach, Globerson, 2023/2024 | ICML 2024 (ML 최상위) | 정답 함수가 그래프를 쓰지 않을 때도 GNN이 그래프를 쓰는 쪽으로 과적합하며, **무한 데이터에서도** 그래프를 무시하는 해를 배운다는 보장이 없다. 경사하강의 암묵 편향으로 설명. 정규 그래프가 더 강건 | 우리 대조 조건(k=1)에서 GNN이 MLP보다 나빠질 수 있다는 것의 이론적 근거 | https://arxiv.org/abs/2309.04332 | 초록 + 검색 요약 |
| 7 | Position: Graph Learning Will Lose Relevance Due To Poor Benchmarks | Bechler-Speicher 외 7인, 2025 | arXiv position (미심사) | "benchmarks should always report the performance of baselines that only process unstructured sets of node features". DeepSets(Empty) 표 1: molhiv 63.78±1.05, molbbbp 64.90±0.72, molbace 51.76±2.85 | 구조 무관 집합 인코더를 **필수 기준선으로 보고하라**는 권고. 우리 MLP 비교군 설계의 정당화 근거 | https://arxiv.org/html/2502.14546v1 | 전문(권고 문장·표 1) |
| 8 | Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs | Zhu, Yan, Zhao, Heimann, Akoglu, Koutra, 2020 | NeurIPS 2020 (ML 최상위) | 이질성(heterophily)이 강한 그래프에서 **MLP가 다수 기존 GNN을 능가**한다. 유효 설계(자기·이웃 임베딩 분리, 고차 이웃, 중간 표현 결합)를 제시 | 구조 정보가 유용해지는 조건을 동질성 수준이라는 명시적 축으로 검증한 선례이나, 우리 축(공급 편중도·묶음 크기)과는 다른 축 | https://proceedings.neurips.cc/paper/2020/file/58ae23d878a47004366189884c2f8440-Paper.pdf | 초록 + 검색 요약 |
| 9 | Investigating the Interplay between Features and Structures in Graph Learning | Castellana, Errica, 2023 | arXiv (미심사) | "six synthetic tasks and evaluate the performance of six models, including structure-agnostic ones". 노드 특징이 라벨과 강하게 상관하지 않는 가정을 풀면 기존 동질성 지표가 부적합해지며, 완전 이질 그래프에서도 성능이 높은 과제를 구성할 수 있다 | 구조 정보의 유용성을 합성 과제 설계로 분리한 선례. 우리 실험 3의 "요약 지표는 같고 구조만 다른 시장" 설계와 같은 발상 | https://arxiv.org/abs/2308.09570 | 초록 |
| 10 | From Local Structures to Size Generalization in Graph Neural Networks | Yehudai, Fetaya, Meirom, Chechik, Maron, 2021 | ICML 2021 (ML 최상위) | 국소 구조가 **그래프 크기에 의존하는** 분포에서는 GNN의 크기 간 일반화가 보장되지 않으며, 작은 그래프에서만 잘 하는 나쁜 전역 최소해가 존재한다 | 우리 진단(주문자 수 4~58, 합 풀링 크기 의존)에 붙일 수 있는 형식적 근거 | https://proceedings.mlr.press/v139/yehudai21a.html | 초록 |
| 11 | Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets | Zhang, Tozzo, Higgins, Ranganath, 2022 | ICML 2022 (PMLR 162) | "both models can suffer from vanishing or exploding gradients"(Deep Sets·Set Transformer). "layer norm can actually hurt performance on tasks with real-valued sets, as its standardization forces potentially unwanted invariance to scalar transformations in set elements" | 집합 인코더의 정규화 선택이 성능을 좌우한다는 근거. 우리가 주문자·공급자 특징을 같은 열에서 함께 표준화한 결함과 같은 계열의 문제 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10465016/ | 전문(해당 절) |
| 12 | How Powerful are Graph Neural Networks? (GIN) | Xu, Hu, Leskovec, Jegelka, 2019 | ICLR 2019 (ML 최상위) | 합 집계는 multiset 전체를 담고(단사), 평균은 원소 종류의 비율을 담고, 최대는 다중도를 버린다 | 합 풀링이 크기 정보를 담는다는 표현력 근거이자, 그 대가로 크기에 비례해 스케일이 변한다는 우리 진단의 배경 | https://arxiv.org/abs/1810.00826 | 초록 + 검색 스니펫 인용(전문 미독) |
| 13 | Modern graph neural networks do worse than classical greedy algorithms in solving combinatorial optimization problems like maximum independent set | Angelini, Ricci-Tersenghi, 2023 | Nature Machine Intelligence (comment, Q1) | 물리기반 비지도 GNN을 최대컷·최대독립집합에서 시험해, 거의 선형 시간의 단순 greedy가 MIS에서 훨씬 좋은 품질의 해를 찾는다 | OR 조합최적화에서 GNN이 단순 방법에 진 공개 사례. 다만 가치함수 근사가 아니라 해 구성 휴리스틱 | https://www.nature.com/articles/s42256-022-00589-y | 초록 + 검색 요약 |
| 14 | Reply to: Modern graph neural networks do worse than classical greedy algorithms | Schuetz, Brubaker, Katzgraber, 2023 | Nature Machine Intelligence (reply, Q1) | 비판이 greedy가 잘 하도록 되어 있는 비대표적 예(희소 그래프 MIS) 하나에 집중했다고 반박 | 음의 결과를 인용할 때 반박도 함께 적어야 한다는 근거 | https://www.nature.com/articles/s42256-022-00590-5 | 검색 요약 |
| 15 | Inability of a graph neural network heuristic to outperform greedy algorithms in solving combinatorial optimization problems | Boettcher, 2023 | Nature Machine Intelligence (comment, Q1) | 같은 논쟁의 또 다른 비판 논평 | 제목만 확인, 내용 미확인 | https://www.nature.com/articles/s42256-022-00587-0 | 서지만 |
| 16 | Hybrid Models for Learning to Branch | Gupta, Gasse, Khalil, Kumar, Lodi, Bengio, 2020 | NeurIPS 2020 (ML 최상위) | GNN의 표현력과 값싼 MLP를 결합해 GPU 없이 최신 기법 대비 해 시간 최대 26% 단축, 학습보다 어려운 문제로 외삽 | 구조 인코더를 루트에서만 쓰고 이후에는 값싼 MLP로 충분하다는 절충 선례. GNN이 틀렸다는 결과는 아니다 | https://arxiv.org/abs/2006.15212 | 초록 + 검색 요약 |
| 17 | Planning with Expectation Models | Wan, Abbas, White, White, Sutton, 2019 | IJCAI 2019 (AI 주요 학회) | "planning with an expectation model is equivalent to planning with a distribution model if the state value function is linear in state features" | **C7의 유일한 형식적 진술.** 우리 g가 비선형 ReLU 헤드인 순간 V(기대 입력) ≠ 기대 V(입력)의 격차가 남는다는 조건을 정확히 준다 | https://arxiv.org/abs/1904.01191 | 초록(핵심 문장 직접 확인) |
| 18 | Learning Optimal Dynamic Matching via Graph Neural Networks | Okada, Noda, Komiyama, Matsushita, 2026 | arXiv (미심사) | "We approximate the value with a graph neural network, train it by temporal-difference learning, and use it in a forward-greedy matching heuristic". 비교군은 즉시 greedy·임계 greedy·patient greedy | 동적 매칭에서 GNN 가치함수 근사를 쓰지만 **MLP·요약 지표 비교군이 없고** 조합 최적화에 넣지 않는다(1차 B5의 [24]와 동일 논문) | https://arxiv.org/abs/2607.28925 | 초록 |
| 19 | HGCN2SP: Hierarchical Graph Convolutional Network for Two-Stage Stochastic Programming | Wu, Zhang, Liang, Cheng, 2025 | arXiv (미심사) | Neur2SP를 인용하는 계층적 GCN 접근 | 제목·서지만 확인. 가치함수 근사의 MLP 대조 여부 미확인 | https://arxiv.org/abs/2511.16027 | 서지만 |
| 20 | Relaxation-Informed Training of Neural Network Surrogate Models | Tsay, 2026 | arXiv (미심사) | 학습 시 LP 완화 격차를 벌점으로 넣어 MILP 해 시간을 크게 줄인다(검색 요약) | 학습·최적화 연결의 문제를 다루지만 **계산 시간** 목적이며 Jensen 격차·분포 이동이 아니다 | https://arxiv.org/abs/2604.22746 | 초록/검색 요약 |
| 21 | Hybrid Value Function Approximation for Solving the Technician Routing Problem with Stochastic Repair Requests | (저자 미확인), 2023 | Transportation Science (SSCI Q1) | 유전 탐색 + GNN 결합 가치함수 근사, 수제 기저함수를 없애는 상태 인코딩(검색 요약) | **페이월 403으로 전문·초록 직접 확인 실패.** MLP/기저함수 대조의 정량 수치 미확인 | https://doi.org/10.1287/trsc.2022.0434 | 서지만(페이월) |

---

## 2. C1~C7 직답

### C1. OR 응용에서 GNN을 가치함수 근사로 쓴 사례와, 요약 지표 기반 근사(MLP·선형) 대비 성과 — **사례는 있으나 "GNN vs 요약 지표 근사"의 정량 대조를 보고한 것은 확인 범위에서 찾지 못했다**

- 동적 매칭: Okada 외(2026)가 사후결정 잔여 그래프의 계속가치를 GNN으로 근사하고 TD 학습한다[18].
  그러나 비교군이 즉시 greedy·임계 greedy·patient greedy뿐이고 **MLP·요약 지표 기반 근사와의 비교가 없다**(초록 직접 확인).
- 기술자 라우팅 + 부품 재고: Transportation Science에 GNN 기반 상태 인코딩으로 수제 기저함수를 없앤 하이브리드 VFA가 있다[21].
  **페이월 403으로 직접 확인 실패** → §5에 확인할 질문을 남긴다. 이것이 C1의 최우선 확인 대상이다.
- 2단 확률계획 계열에서 GCN을 쓰는 HGCN2SP[19]가 있으나 서지만 확인했다.
- 1차 조사에서 확인한 바(B5): 그래프가 고정되면 GNN 최적화가 밀집망 최적화와 동등하다는 진술,
  Graph4BiLO의 종류별 평균 풀링 + 선형 읽기.
- **판정**: "GNN VFA가 요약 지표 MLP보다 얼마나, 어떤 조건에서 나은가"를 정량 보고한 OR 논문은 확인하지 못했다.
  우리 실험 3은 그 빈 칸에 음의 방향으로 들어가는 자료이며, 갭 주장으로 쓸 수 있는 가장 단단한 부분이다.
  확인한 쿼리: `"graph neural network" "value function approximation" inventory routing compared to feature-based approximation`,
  `graph neural network value function approximation dynamic matching ride-hailing compared MLP aggregate features`(OpenAlex),
  `graph neural network value function approximation dynamic vehicle routing inventory compared to handcrafted features`.

### C2. 구조 정보가 요약 통계보다 유용해지는 조건을 명시적으로 실험 설계에 넣어 검증한 연구 — **GNN 일반 문헌에는 있고, OR 응용에는 확인하지 못했다**

- 동질성/이질성 축: Zhu 외(NeurIPS 2020)가 동질성 수준을 축으로 두고, 이질성이 강할 때 **MLP가 여러 GNN을 능가**함을 보인다[8].
- 특징·구조 정보량 분리: Castellana & Errica(2023)는 합성 과제 6개를 만들어 구조 무관 모델을 포함해 비교하고,
  노드 특징이 라벨과 강하게 상관한다는 가정을 풀면 기존 동질성 지표가 부적합해진다고 보고한다[9].
- 그래프 분포 축: Bechler-Speicher 외(ICML 2024)는 그래프 분포별로 비교하고 정규 그래프가 더 강건하다고 보고한다[6].
- **우리 축(공급 편중도, 대체 공급자 수, 주문당 품목 수)으로 이런 ablation을 한 사례는 찾지 못했다.**
  즉 C2의 발상 자체는 GNN 문헌의 표준 관행이고, 축을 시장 구조 파라미터로 잡은 것이 우리 몫이다(추론).

### C3. GNN과 MLP를 공정하게 비교하는 프로토콜의 표준, 우리 방식에 빠진 것 — **표준은 Errica 외(ICLR 2020)이고, 우리에게 빠진 것이 네 가지 있다**

Errica 외의 프로토콜(전문 직접 확인)[5]:
1. 구조 무관 기준선을 **반드시** 포함. 화학: 노드 특징 합 풀링 + 1층 MLP. 소셜: 1층 MLP → 합 풀링 → 1층 MLP.
2. 10-fold 외부 교차검증(모델 평가) + 내부 90/10 holdout(모델 선택)의 **이중 분리**.
3. 모델 선택 후 각 fold에서 **최종 3회 재학습**.
4. 하이퍼파라미터 그리드 32~72 구성. **파라미터 수를 맞추지 않고**, 그리드 크기도 하이퍼파라미터 개수에 따라 모델마다 다르다.

Bechler-Speicher 외(2025)는 여기에 "구조 없는 집합 인코더 기준선을 항상 보고하라"를 명시적 권고로 올린다[7].

우리 방식(`exp3_diag.json` 기준: 튜닝 seed 1개, 반복 0–71 학습 / 72–89 검증, wd 원소 2개)에 빠진 것:
- **(a) 외부 교차검증·시드 반복이 없다.** 학습/검증 분할이 1개이고 최종 재학습도 1회다. 칸별 R² 차이의 표준오차를 낼 수 없다.
  conc1_k1의 GNN 검증 R² 0.0073 같은 극단값이 분할 운인지 구분할 수 없다.
- **(b) 하이퍼파라미터 그리드가 weight decay 2개로 매우 작다.** Errica의 32~72 구성과 자릿수가 다르다.
  게다가 GNN 학습이 칸당 약 300초, MLP가 약 1.4초이므로 **같은 예산이 같은 탐색 폭을 뜻하지 않는다**.
- **(c) 파라미터 수 동등화를 보고하지 않았다.** 단, Errica도 파라미터 수를 맞추지 않으므로 이것은 표준 위반이 아니다[5].
  CLAUDE.md의 "차이가 나면 해석 전에 구현 차이(파라미터 수, 학습량)를 먼저 점검한다"는 규칙은 표준보다 엄격하다.
- **(d) 입력 정보량 동등화는 했으나(같은 시장 요약 지표 + 편중도 요약 비교군), 입력 전처리 동등화는 깨졌다.**
  주문자·공급자 특징이 같은 열을 공유하고 함께 표준화되어 공급자 수량 최대 |z|가 10~11이다(`exp3_diag.json`).
  Zhang 외(ICML 2022)는 집합 데이터에서 정규화 선택이 성능을 좌우한다고 보고한다[11].
  **즉 현재 결과는 "구조 정보가 쓸모없다"가 아니라 "GNN 쪽 전처리가 불리했다"로도 읽히며, 이 상태로는 공정 비교라고 주장할 수 없다.**

### C4. 가변 노드 수·활성 마스크의 표준 처리와 그 처리가 예측 성능에 주는 영향 — **부분적으로 답을 찾았다**

- 표준 처리는 패딩 + 마스킹 + 순서불변 풀링이다. 우리 구현(최대 자리 + 활성 마스크)은 이 계열이다.
  다만 이 조합의 성능 영향을 OR 맥락에서 정면으로 보고한 논문은 찾지 못했다.
- 풀링 선택의 표현력 대가: 합은 multiset 전체를 담고(단사), 평균은 비율만 담고, 최대는 다중도를 버린다[12].
  따라서 합 풀링은 **크기 정보를 담는 대신 크기에 비례해 스케일이 변한다**(우리 진단과 직결).
- 크기 일반화 실패의 형식적 근거: Yehudai 외(ICML 2021)는 국소 구조가 그래프 크기에 의존하는 분포에서
  GNN의 크기 간 일반화가 보장되지 않고, 작은 그래프에서만 잘 하는 나쁜 전역 최소해가 존재함을 증명한다[10].
  우리 상태의 주문자 수는 4~58로 14배 범위이므로 이 조건에 해당한다(추론).
- 집합 인코더의 정규화: Zhang 외(ICML 2022)는 Deep Sets·Set Transformer가 기울기 소실·폭발을 겪고,
  layer norm이 실수값 집합 과제에서 성능을 해칠 수 있다고 보고하며 set norm을 제안한다[11].
- **미해결**: 활성 마스크 방식이 가변 참여자 수 환경에서 예측 성능에 주는 영향을 직접 측정한 OR 논문은 확인하지 못했다.
  확인한 쿼리: `permutation invariant neural network variable number of agents value function approximation operations masking`(OpenAlex),
  `deep sets sum pooling scale with set size normalization variable set size masking padding`.

### C5. 음의 결과 보고 — **있다. 5편 확정.** 별도 절(§3) 참조

### C6. Neur2RO 계보에서 풀링 가중치가 연속 결정변수인 사례 — **있다. Neur2BiLO가 연속 상위 결정변수를 포함한 합 풀링을 MILP에 넣는다.** 판정은 §4 참조

확인한 사실(모두 전문 직접 확인):
- **Neur2SP**[1]: 시나리오 임베딩을 **평균 집계**하고, 집계는 결정과 무관하며 잠재 표현만 MIP에 넣는다.
  "Note that the embedding networks can be arbitrarily complex as only the latent representation is embedded into the approximate MIP."
  → 풀링 가중치는 결정변수가 아니다.
- **Neur2RO**[2]: 1단 변수별 임베딩망과 작은 값망을 MILP로 표현하고, 시나리오 임베딩은 사전 계산한다.
  "the scenario embeddings can be precomputed via a forward pass for each scenario, i.e., no MILP representation is needed."
  → **집계되는 항 자체가 결정변수의 함수이고 그 집계가 MILP 안에서 일어난다.** 다만 실험의 1단 결정은 이진·정수 중심이다.
- **Neur2BiLO**[3]: "Embedding the set of variable features using a set-based architecture ... summing up the resulting n variable embeddings",
  "akin to the DeepSets approach". 표 1에서 상위 결정변수 유형에 **연속(C)** 이 있다(donor-recipient).
  → **합 풀링 항이 연속 결정변수에 의존하는 사례가 계보 안에 이미 있다.**
- **SEQUOIA / coRMAB**[4]: 학습 Q망을 매 기간 MILP에 넣어 조합 행동을 고르는 **다기간 루프**를 이미 한다.
  "we embed the trained Q-network into the MILP formulations of our constraints and use its output as our objective."
  행동은 이진 벡터이고 Q망은 2층×32 단일 MLP로 풀링·개체별 분해가 없다.
- Neur2RO의 피인용(Semantic Scholar 3건: Neur2BiLO, Optimization Over Trained Neural Networks: Taking a Relaxing Walk, L2O-CCG)과
  Neur2SP의 피인용 61건, Neur2BiLO의 피인용 10건을 훑었다. OpenAlex는 Neur2RO의 피인용을 0건으로 기록한다(색인 누락).
  피인용 목록 안에서 **풀링 가중치를 연속 확률 결정변수로 둔 사례는 Neur2BiLO 외에 추가로 찾지 못했다.**

### C7. V(기대 입력) vs 기대 V(입력)의 Jensen 격차, 학습·최적화 입력 분포 불일치 — **계보 안에서 이 문제를 인지·처리한 연구를 찾지 못했다. 이것이 확인된 빈 칸이다**

직접 확인한 부재(전문 확인):
- **Neur2SP**: Jensen 부등식이나 집계 근사의 편향에 대한 이론적 취급이 **없다**. 목적함수 격차를 경험적으로만 보고한다.
  분포 불일치에 관한 유일한 언급은 "Although Ψ1 is trained using K scenarios, once the networks are trained,
  they can be used with any (potentially much larger) finite number of scenarios."이고, 그 편향은 논의하지 않는다[1].
- **Neur2RO**: Jensen형 격차도, 학습·최적화 입력 분포 불일치도, 신뢰영역도 다루지 않는다[2].
- **Neur2BiLO**: 분포 이동·신뢰영역을 다루지 않는다. 학습된 모델이 안전장치 없이 일반화된다고 가정한다[3].
- **SEQUOIA**: Jensen 근사나 학습·추론 입력 격차를 다루지 않는다. 학습과 추론이 같은 MILP 임베딩을 쓴다[4].

가장 가까운 형식적 진술은 계보 밖(RL)에 있다:
- Wan 외(IJCAI 2019): "planning with an expectation model is equivalent to planning with a distribution model
  **if the state value function is linear in state features**"[17].
  → 우리 설계로 옮기면, z = Σ p_i h_i에 대해 미래 가치를 g(z)로 두는 것이 기대 V와 일치하는 것은 **g가 선형일 때뿐**이다.
  작은 ReLU 헤드 g를 쓰는 순간 Jensen 격차가 남고, 그 부호는 g의 볼록성 방향에 달린다.
  학습을 p 원소가 0 또는 1인 실현 그래프로 하고 최적화를 p 원소가 0과 1 사이인 점에서 하면
  **격차 + 지지집합 밖 외삽**이 겹친다.
- 인접 장치: 1차 B6에서 확인한 신뢰영역 계열(OptiCL의 볼록껍질, Shi 외 IJOC 2024의 isolation forest)이 입력 공간 외삽을 막지만,
  "학습은 정수 실현, 최적화는 소수 확률"이라는 특정 불일치를 다루지는 않는다.
- Tsay(2026)[20]는 학습 시 LP 완화 격차를 벌점화하지만 목적이 **해 시간**이며 근사 편향이 아니다.

확인한 쿼리(결과 없음): `"Jensen gap" neural network surrogate expected value function stochastic programming embedded optimization bias`,
`neural network surrogate trained on integral inputs evaluated at fractional relaxed inputs optimization invalid extrapolation embedded MILP`,
`graph neural network probabilistic graph expected embedding node existence probability evaluate network at expected input bias optimization`,
`approximate dynamic programming "post-decision state" learned value function evaluated at expected state bias nonlinear approximation certainty equivalent`,
`learned neural surrogate expected value decision-dependent probability distribution embedded mixed integer program`.
→ 이 다섯 쿼리 모두 계보 내 직접 처리 사례를 내놓지 않았다.

---

## 3. C5 음의 결과 목록 (우리 실험 3 v1의 방어 근거)

| 근거 | 무엇을 보고했는가 | 우리 결과와의 대응 | 인용 시 주의 |
|---|---|---|---|
| Errica 외, ICLR 2020[5] | 9개 벤치마크 중 **D&D, PROTEINS, ENZYMES 3개에서 어떤 GNN도 구조 무관 기준선을 넘지 못했다**. 기준선은 노드 특징 합 풀링 + MLP | 우리 기준선(시장 요약 지표 MLP)이 구조적으로 그들의 구조 무관 기준선과 같다. GNN이 못 넘은 것이 이 문헌에서 **이례가 아니다** | 저자들의 결론은 "구조가 쓸모없다"가 아니라 "아직 활용되지 못했다(not able to fully exploit ... yet)"다. 우리도 그렇게 써야 한다 |
| Bechler-Speicher 외, ICML 2024[6] | 정답 함수가 그래프를 쓰지 않을 때 GNN은 그래프를 쓰는 쪽으로 과적합하고, **무한 데이터에서도** 그래프 무시 해를 배운다는 보장이 없다 | 우리 대조 조건 k=1에서 "차이가 없어야 한다"는 기대가 **오히려 GNN이 나빠지는 방향으로 깨질 수 있다**는 이론 근거 | 경사하강 암묵 편향에 기반한 결과이며 특정 설정 가정이 있다(전문 미독, §5) |
| Bechler-Speicher 외, position 2025[7] | DeepSets(빈 그래프) 기준선이 OGB에서 경쟁력: molhiv 63.78±1.05, molbbbp 64.90±0.72, molbace 51.76±2.85. 구조 무관 기준선 보고를 권고 | 우리가 MLP 비교군을 둔 설계가 **최신 권고와 일치한다**는 근거 | position paper이며 심사를 통과한 정식 논문이 아니다(미심사 표시 필요) |
| Zhu 외, NeurIPS 2020[8] | 이질성이 강한 그래프에서 **MLP가 다수 기존 GNN을 능가**한다 | GNN이 MLP에 지는 조건이 그래프 성질로 특성화된 선례 | 노드 분류 과제이고 회귀·가치함수 근사가 아니다 |
| Angelini & Ricci-Tersenghi, Nature MI 2023[13] | 최대독립집합에서 거의 선형 시간 greedy가 물리기반 GNN보다 훨씬 나은 해를 찾는다 | OR 조합최적화에서 GNN이 단순 방법에 진 공개 사례 | 원저자 반박[14]과 또 다른 논평[15]이 있는 **논쟁 중인 사안**이다. 반박을 함께 적지 않으면 인용이 약해진다 |
| Gupta 외, NeurIPS 2020[16] | GNN을 루트에서만 쓰고 이후는 값싼 MLP로 대체해 GPU 없이 최대 26% 시간 단축 | 구조 인코더가 항상 필요하지는 않다는 절충 선례 | **GNN이 실패했다는 결과가 아니다.** 하이브리드가 더 좋다는 결과이므로 음의 결과로 분류하면 과장이다 |

**우리 결과를 이 목록에 놓을 때의 정확한 서술(권고)**:
"GNN이 구조 무관 기준선을 넘지 못하는 것은 그래프 학습 문헌에서 반복 보고된 현상이며[5][7][8], 그 원인 후보로는
그래프 과의존[6], 크기 일반화 실패[10], 집합 인코더 정규화[11]가 있다. 우리 v1 결과는 이 중 **크기 의존 합 풀링과 정규화 결함**을
진단으로 특정했으므로, 현재 수치는 '이 규모에서 구조 정보가 무효'의 증거가 아니라 **구현 결함이 섞인 미결 상태**로 보고해야 한다."
이는 CLAUDE.md의 "결과가 예상과 다르면 먼저 구현 오류를 양쪽 모두에 대해 같은 기준으로 점검"과 일치한다.

---

## 4. C6 판정: Neur2RO 계보에서 우리 대안 설계에 남는 차이

### 구조 자체는 새롭지 않다 (확정)

우리 대안 설계 = 노드 임베딩 h_i 사전 계산 → z = Σ_i p_i h_i (p_i는 MILP 안의 구간선형 재참여확률) → 작은 ReLU 헤드 g만 임베딩.

- "결정 무관 부분 사전 계산 + 결정 의존 부분만 임베딩"은 **Neur2RO의 명시적 설계**다[2].
- "집합 원소별 임베딩을 합으로 풀링하고 그 항이 결정변수에 의존"은 **Neur2BiLO의 DeepSets식 합 풀링**이며,
  상위 결정변수가 **연속**인 응용까지 포함한다[3].
- 우리 z = Σ p_i h_i는 h_i가 상수이므로 p_i에 대해 **선형**이다. Neur2RO·Neur2BiLO는 집계 항이 결정변수의 **비선형 망**이다.
  즉 우리 구조는 그 계보의 **특수형이자 더 쉬운 경우**다. "우리가 더 어려운 구조를 다뤘다"고 주장할 수 없다.

### 남는 차이 세 가지와 각각의 통과 가능성

| 차이 | 계보 안에 있는가 | OR/OM 리뷰어의 "왜 중요한가" 통과 가능성 | 솔직한 판정 |
|---|---|---|---|
| (1) 풀링 가중치가 결정변수의 **구간선형 함수**(p_i = PWL(잉여율)) | 정확히 같은 것은 못 찾았으나, PWL을 신경망 입력 앞에 합성하는 것은 addGenConstrPWL로 곧바로 되는 표준 조작이다 | **낮다.** 합성 한 단계 추가는 방법론적 신규성으로 인정받기 어렵다 | **통과 못 한다.** 이것을 기여로 내세우면 안 된다 |
| (2) 참여자 집합이 결정에 따라 전이하는 **다기간 ADP 루프** | 학습 Q를 MILP에 넣고 매 기간 푸는 루프는 SEQUOIA(ICLR 2025)[4]와 1차 B2의 4편이 이미 한다. 그러나 **풀링 + 연속확률 가중치 + ADP 루프의 조합**은 확인 범위에서 없다 | **중간.** 조합 신규성이므로 그 조합이 **새로운 어려움을 만든다**는 것을 보여야만 통과한다. "처음이다"만으로는 부족하다 | 조건부 통과. 어려움의 증거가 필요하고, 그 증거의 자연스러운 후보가 (3)이다 |
| (3) **Jensen 격차**: 학습은 실현 그래프(p 원소 0/1), 최적화는 소수 확률 | **계보 어디에도 없다**(Neur2SP·Neur2RO·Neur2BiLO·SEQUOIA 전문 확인). 형식 조건은 계보 밖 RL에 있다(Wan 외 2019: 헤드가 선형일 때만 무편향)[17] | **가장 높다.** 계보가 공유하는 설계의 **알려지지 않은 위험**을 특정하고 크기를 재는 것은 OR/OM 리뷰어가 받아들이는 형태의 기여다 | 여기에 기여를 걸어야 한다. 단, **격차를 실제로 측정해야** 한다. 측정 없이 위험만 적으면 통과 못 한다 |

### 종합 판정 (솔직하게)

- 우리 대안 설계를 **방법론적 신규성**으로 내세우면 **통과하지 못한다.** Neur2RO[2]와 Neur2BiLO[3]가
  "사전 계산 + 결정 의존 풀링 + 작은 헤드 임베딩"을 이미 하고 있고, 우리 것은 그 중 선형인 특수형이다.
  리뷰어가 Neur2BiLO 한 편만 들고 와도 "구조는 이미 있다"가 성립한다.
- 통과 가능한 형태는 두 가지다(추론, 우리 판단):
  1. **위험 특정 + 정량화**: 이 계보의 집계-임베딩 설계가 확률 가중치로 확장될 때 발생하는 Jensen 격차를 정의하고,
     우리 환경에서 (i) 헤드가 선형일 때와 비선형일 때, (ii) 실현 그래프 학습 vs 소수 확률 학습의 차이를 실측해 보고한다.
     Wan 외[17]의 선형성 조건이 그 논증의 유일한 기존 고리이고, 나머지는 우리가 직접 논증해야 한다.
  2. **변환 오차 비교**: 현재 방식(1차 분해 c_i)과 풀링 임베딩 방식의 **변환 오차**를 같은 시뮬레이션 기준치로 비교한다.
     1차 조사 §4가 확인한 갭 (a) "분해 오차를 시뮬레이션 기준치로 정량화"와 곧바로 이어진다.
- **하지 말아야 할 주장**: "풀링 가중치가 결정변수인 구조는 우리가 처음이다"(Neur2BiLO 반례),
  "학습 V를 MILP에 넣고 매 기간 푸는 것은 우리가 처음이다"(SEQUOIA·1차 B2 반례).

---

## 5. 미해결·확인 불가 (페이월 목록과 확인할 질문)

| 항목 | 상태 | 확인할 구체적 질문 |
|---|---|---|
| Hybrid Value Function Approximation for the Technician Routing Problem, Transportation Science[21] | **페이월 403**(INFORMS) | GNN 상태 인코딩을 수제 기저함수 기반 VFA와 직접 비교했는가? 개선폭의 수치는? 어떤 조건에서 차이가 커지는가? **C1의 최우선 확인 대상** |
| Errica 외 ICLR 2020[5] 부록의 모델별 파라미터 수 | 전문은 읽었으나 파라미터 수 표는 확인 못 함 | GNN과 구조 무관 기준선의 파라미터 수 차이를 어떻게 보고하는가? 우리 C3 (c) 항목의 판단 근거 |
| Bechler-Speicher 외 ICML 2024[6] 전문 | 초록·검색 요약만 | 그래프 과의존이 **회귀·가치함수 근사**에서도 성립하는가, 노드 분류에 한정된 결과인가? 우리 k=1 대조 조건에 인용할 수 있는지가 여기에 달려 있다 |
| GIN[12]의 합/평균 집계 진술 | 검색 스니펫 인용, 전문 미독 | 해당 진술의 정확한 절·문장. 합 풀링이 크기에 비례해 스케일이 변한다는 것을 논문이 직접 말하는가, 아니면 우리 추론인가 |
| HGCN2SP[19] | 서지만 | GCN을 가치함수 근사 자체에 쓰는가 시나리오 선택에만 쓰는가? MLP 대조가 있는가? |
| Yehudai 외 ICML 2021[10] 전문 | 초록만 | "국소 구조가 크기에 의존"의 형식 정의가 우리 상태(주문자 수 4~58, 편중도 고정)에 해당하는가? |
| Neur2RO 피인용의 색인 누락 | OpenAlex 0건, Semantic Scholar 3건 | 다른 색인(Google Scholar 등)으로 피인용을 다시 훑을 필요가 있다. 현재 C6 판정은 S2 3건 + Neur2SP 61건 + Neur2BiLO 10건 범위에 근거한다 |
| 분포적 제약학습(결정 의존 분포 + NN 임베딩, Expert Systems with Applications 2023) | **페이월 403**(ScienceDirect), DOI 특정 실패 | 결정변수가 불확실성의 분포를 바꿀 때 NN 임베딩이 Jensen형 편향을 어떻게 다루는가? C7의 잠재적 반례이므로 확인 필요 |

---

## Coverage Status

| 질문 | 상태 | 비고 |
|---|---|---|
| C1 | **needs follow-up** | 사례는 특정했으나 GNN vs 요약 지표 근사 정량 대조는 못 찾음. TS 논문[21] 페이월이 병목 |
| C2 | **done** | GNN 일반 문헌에 축 기반 ablation 선례 3편 확인. OR 응용에는 없음 |
| C3 | **done** | Errica 프로토콜 전문 확인. 우리에게 빠진 것 4개 특정 |
| C4 | **부분 done** | 풀링 표현력·크기 일반화·집합 정규화 근거 확보. 마스킹 방식의 성능 영향을 직접 측정한 OR 논문은 없음 |
| C5 | **done** | 음의 결과 5편 + 절충 1편. §3에 별도 절 |
| C6 | **done** | Neur2SP·Neur2RO·Neur2BiLO·SEQUOIA 전문 확인. 판정: 구조 신규성 없음, 기여는 C7로 옮겨야 함 |
| C7 | **done (부재 확인)** | 계보 4편 전문에서 Jensen 격차·분포 불일치 취급 없음. 형식 조건은 Wan 외 2019뿐 |

**직접 확인한 것**: [1][2][3][4][5][7][11]의 전문 해당 절, [17]의 핵심 문장, [18]의 초록,
Neur2SP/Neur2RO/Neur2BiLO의 피인용 목록(OpenAlex + Semantic Scholar API).
**불확실한 것**: [6][8][10][12][13]은 초록·검색 요약 수준이며 전문 미독. [15][19][21]은 서지만.
**완료하지 못한 것**: TS 기술자 라우팅 논문[21]과 분포적 제약학습 논문의 페이월 접근.

---

## 6. 출처 목록

1. Dumouchelle, Patel, Khalil, Bodur — Neur2SP: Neural Two-Stage Stochastic Programming (NeurIPS 2022) — https://arxiv.org/abs/2205.12006
2. Dumouchelle, Julien, Kurtz, Khalil — Neur2RO: Neural Two-Stage Robust Optimization (ICLR 2024) — https://arxiv.org/html/2310.04345v2
3. Dumouchelle, Julien, Kurtz, Khalil — Neur2BiLO: Neural Bilevel Optimization (NeurIPS 2024) — https://arxiv.org/html/2402.02552v2
4. Xu, Wilder, Khalil, Tambe — Reinforcement learning with combinatorial actions for coupled restless bandits (ICLR 2025) — https://arxiv.org/html/2503.01919v1
5. Errica, Podda, Bacciu, Micheli — A Fair Comparison of Graph Neural Networks for Graph Classification (ICLR 2020) — https://arxiv.org/abs/1912.09893
6. Bechler-Speicher, Amos, Gilad-Bachrach, Globerson — Graph Neural Networks Use Graphs When They Shouldn’t (ICML 2024) — https://arxiv.org/abs/2309.04332
7. Bechler-Speicher 외 — Position: Graph Learning Will Lose Relevance Due To Poor Benchmarks (2025, 미심사) — https://arxiv.org/html/2502.14546v1
8. Zhu, Yan, Zhao, Heimann, Akoglu, Koutra — Beyond Homophily in Graph Neural Networks (NeurIPS 2020) — https://proceedings.neurips.cc/paper/2020/file/58ae23d878a47004366189884c2f8440-Paper.pdf
9. Castellana, Errica — Investigating the Interplay between Features and Structures in Graph Learning (2023, 미심사) — https://arxiv.org/abs/2308.09570
10. Yehudai, Fetaya, Meirom, Chechik, Maron — From Local Structures to Size Generalization in Graph Neural Networks (ICML 2021) — https://proceedings.mlr.press/v139/yehudai21a.html
11. Zhang, Tozzo, Higgins, Ranganath — Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets (ICML 2022) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10465016/
12. Xu, Hu, Leskovec, Jegelka — How Powerful are Graph Neural Networks? (ICLR 2019) — https://arxiv.org/abs/1810.00826
13. Angelini, Ricci-Tersenghi — Modern graph neural networks do worse than classical greedy algorithms ... (Nature Machine Intelligence 2023) — https://www.nature.com/articles/s42256-022-00589-y
14. Schuetz, Brubaker, Katzgraber — Reply to: Modern graph neural networks do worse ... (Nature Machine Intelligence 2023) — https://www.nature.com/articles/s42256-022-00590-5
15. Boettcher — Inability of a graph neural network heuristic to outperform greedy algorithms ... (Nature Machine Intelligence 2023) — https://www.nature.com/articles/s42256-022-00587-0
16. Gupta, Gasse, Khalil, Kumar, Lodi, Bengio — Hybrid Models for Learning to Branch (NeurIPS 2020) — https://arxiv.org/abs/2006.15212
17. Wan, Abbas, White, White, Sutton — Planning with Expectation Models (IJCAI 2019) — https://arxiv.org/abs/1904.01191
18. Okada, Noda, Komiyama, Matsushita — Learning Optimal Dynamic Matching via Graph Neural Networks (2026, 미심사) — https://arxiv.org/abs/2607.28925
19. Wu, Zhang, Liang, Cheng — HGCN2SP: Hierarchical Graph Convolutional Network for Two-Stage Stochastic Programming (2025, 미심사) — https://arxiv.org/abs/2511.16027
20. Tsay — Relaxation-Informed Training of Neural Network Surrogate Models (2026, 미심사) — https://arxiv.org/abs/2604.22746
21. (저자 미확인) — Hybrid Value Function Approximation for Solving the Technician Routing Problem with Stochastic Repair Requests (Transportation Science) — https://doi.org/10.1287/trsc.2022.0434
