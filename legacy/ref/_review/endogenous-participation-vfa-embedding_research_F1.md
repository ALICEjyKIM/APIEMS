# 3차 조사 F1: 구매 관리·공급망 위험 — 상호의존의 종류를 축으로

조사자: researcher 서브에이전트 / 작성 2026-09-28
과제: ref/literature-review.md "3차 조사 범위 (주제 F)" 중 **F1a~F1d**
축: 개체(공급자·고객) 간 **상호의존의 종류** = 대체형 / 보완형 / 혼재 / 해당 없음

우리 설정(요약): 5PL 미들마일 B2B 플랫폼. 플랫폼은 공급자를 **고용·소유하지 않는다.** 매 기간 거래잉여를
주문자·공급자에게 나누고, 그 배분이 **다음 기간 재참여 확률**을 정한다. 주문은 다품목 all-or-nothing.
품목당 공급자 2명, 빈 공급자 자리는 매 기간 확률 0.25로만 충원(빈자리 평균 4기간).

---

## 0. 방법과 제약 (먼저 밝힘 — 결과 해석에 영향)

- **지시받은 OpenAlex API는 이번 세션에서 사용 불가**였다. 첫 호출에서 즉시 반환:
  Rate limit exceeded / dailyRemainingUsd 0 / retryAfter 54374
  (API 키 없는 공유 IP의 일일 무료 예산 소진, UTC 자정 리셋). 따라서 **"Kraljic 1983의 OpenAlex id를 찾아
  피인용을 긁는다"는 지시된 주 작업을 OpenAlex로 수행하지 못했다.**
  대체 수단: (a) **Semantic Scholar Graph API** 로 Kraljic 정량화 논문의 **피인용 42건 목록**을 1회 확보
  (이후 호출은 모두 HTTP 429), (b) **Crossref REST API** (api.crossref.org, 키 불필요) 로
  출판사 등록 초록(JATS abstract) 전문을 확보. Crossref가 이번 조사의 실질적 주 수단이었다.
- **읽은 수준의 정의** (표에서 사용):
  - 초록(전문) = Crossref/SSRN에 저자·출판사가 등록한 **초록 전체**를 직접 읽음. 초록 조각·검색 스니펫이 아님.
  - 서지만 = 제목·저자·게재지·연도만 확인. 내용 주장 없음.
  - 검색 스니펫 = 검색엔진 요약만 확인 → 내용 주장은 신뢰도 low로 표기.
- **페이월**: sciencedirect.com, pubsonline.informs.org는 WebFetch에 **HTTP 403**을 반환한다(3건 시도 전부 실패).
  따라서 **본문(전문)은 한 편도 읽지 못했다.** 본문 확인이 필요한 질문은 4절에 정리했다.
- **WebSearch의 SEO 오염(기록)**: F1a·F1c 각도의 일반 웹 검색은 상위 결과 대부분이 실무 블로그·소프트웨어
  비교 페이지였다 — artofprocurement.com, planergy.com, fractory.com, procurementtactics.com, umbrex.com,
  badgerlogistics.com, arktms.com, atsinc.com, redwoodlogistics.com, zapro.ai, simfoni.com, wcvendors.com,
  cs-cart.com, flxpoint.com, marketplacer.com, extensiv.com, lawinsider.com, jdsupra.com 등.
  **이들은 증거 표에 넣지 않았다.** 특히 "마켓플레이스 판매자 유지 인센티브"(F1c) 각도는
  학술 결과가 **0건**이고 전부 커미션 요율 마케팅 글이었다(3.3절 쿼리 목록 참조).

---

## 1. 증거 표

등급 표기: Q1/Q2는 통상적 평판·색인 기준의 대략적 표기이며 사분위를 개별 확인하지는 않았다.

| # | 제목 | 저자·연도 | 게재지·등급 | 모형 종류 | 유지 수단 | **상호의존 종류** | 우리와의 차이 (한 문장) | URL | 읽은 수준 |
|---|------|-----------|-------------|-----------|-----------|-------------------|--------------------------|-----|-----------|
| 1 | Independence of Capacity Ordering and Financial Subsidies to Risky Suppliers | Babich (2010) | M&SOM (SSCI, OM 주요) | 동적·확률적 주기검토 DP: 용량예약 + 보조금 **동시 결정**, 공급자 재무상태는 일반 firm-value 모형 | **금전 보조금**(subsidize-up-to 구조)으로 공급자 도산 방지 | **해당 없음** (공급자 1명·단일 품목) | 소유하지 않은 공급자를 돈으로 살려두는 DP라는 점은 같으나, 공급자가 1명이라 공급자 간 상호의존이 정의되지 않고 결정변수가 배분 몫이 아니라 보조금 액수이며 학습 VFA가 없다 | https://doi.org/10.1287/msom.1090.0284 | 초록(전문) |
| 2 | Long-Term Contracts Under the Threat of Supplier Default | Swinney & Netessine (2009) | M&SOM (SSCI, OM 주요) | 2기간 계약 게임, **동일한 공급자 2명**, 공급자는 외부자본 없음 → 구매자 매출이 도산 여부를 결정 | **장기계약**(물량·수익 보장) | **대체형** (동일 공급자 2명이 같은 사업을 두고 경쟁; 원가는 공통항 + 고유항) | "구매자가 준 돈이 공급자 생존을 정한다"는 고리는 같으나 2기간 유한 게임이고 동일 품목 대체 공급자만 있어 묶음 보완성이 없다 | https://doi.org/10.1287/msom.1070.0199 | 초록(전문) |
| 3 | Buyer Intermediation in Supplier Finance | Tunca & Zhu (2018) | Management Science (SSCI Q1) | 게임이론 + 중국 온라인 소매업체 데이터 구조추정 + 반사실 분석 | **조기 지불·금융 중개**(구매자가 공급자 금융을 중개해 이자율·도매가 인하) | **해당 없음** (dyadic 채널) | 구매자가 소유하지 않은 공급자에게 금전 수단을 쓰는 실증 사례지만 채널 조정 문제이고 참여자 집합 전이·다품목 묶음이 없다 | https://doi.org/10.1287/mnsc.2017.2863 | 초록(전문) |
| 4 | Competition and Diversification Effects in Supply Chains with Supplier Default Risk | Babich, Burnetas, Ritchken (2007) | M&SOM (SSCI, OM 주요) | Stackelberg 게임(공급자가 도매가 선도) + 뉴스벤더 소매업체, 도산 **상관관계** | 없음 (경쟁·분산 활용) | **대체형** (초록 명시: cheapest supplier 대 order diversification 상충; 도산 상관이 **낮으면** 공급자 간 경쟁이 약해져 도매가가 오름) | 공급자 상호의존을 **경쟁(대체)** 으로만 모형화하며 "소매업체는 도산이 강상관인 공급자를 선호한다"는 결론은 우리 보완형 방향과 반대다 | https://doi.org/10.1287/msom.1060.0122 | 초록(전문) |
| 5 | On the Value of Mitigation and Contingency Strategies for Managing Supply Chain Disruption Risks | Tomlin (2006) | Management Science (SSCI Q1) | 단일 품목, 신뢰/불신뢰 공급자 2명, 용량 제약·볼륨 유연성; 최적 전략 분류(재고 완화 / 신뢰 공급자 single sourcing / 수동 수용) | 없음 (재고·소싱 완화) | **대체형** (두 공급자가 같은 수요를 나눠 충족) | **single sourcing을 최적 전략의 한 결과로 도출하는 정량 모형**이지만 "공급자 이탈"이 아니라 가동률(uptime)이고 현재 결정이 미래 공급자 집합을 바꾸지 않는다 | https://doi.org/10.1287/mnsc.1060.0515 | 초록(전문) |
| 6 | Supply Disruptions, Asymmetric Information, and a Backup Production Option | Yang, Aydin, Babich, Beil (2009) | Management Science (SSCI Q1) | 메커니즘 설계(최적 계약 메뉴), 공급자 신뢰도가 사적 정보 | 계약 메뉴(페널티 대 백업 생산) | **해당 없음** (공급자 1명, 유형 2개) | 정보 비대칭 하 계약 설계로, 참여자 집합 동태·묶음·배분 몫이 없다 | https://doi.org/10.1287/mnsc.1080.0943 | 초록(전문) |
| 7 | **Driving Supply to Marketplaces: Optimal Platform Pricing When Suppliers Share Inventory** | Martinez-de-Albeniz, Pinto, Amorim (2022) | M&SOM (SSCI, OM 주요) | **최적제어**(한정 재고 + 확률적 수요 과정); 공급자 참여를 재고수준·수요율·잔여기간·수수료구조의 함수로 특성화하고 그에 맞춰 플랫폼이 수수료 결정 | **수수료(커미션) 구조** — 소유하지 않은 공급자의 **참여를 유도** | **해당 없음** (dyadic: 공급자 1명–플랫폼 1개) | **F1c에 가장 가까운 선례.** 그러나 공급자가 1명이라 공급자 간 상호의존·조합 배정·묶음이 전혀 없고 참여가 확률적 재참여가 아니라 결정론적 수락 조건이며 학습 VFA가 없다 | https://doi.org/10.1287/msom.2022.1105 | 초록(전문) |
| 8 | To stop or not, that is the question: When should a buyer hit the brakes for supply base rationalization? | Zhang & Katok (2025) | Omega (SSCI Q1) | **최적정지 + 이산 볼록 해석**: control-limit·control-band·one-step look-ahead 정책의 충분조건 + 행동실험 + 구조추정 | 없음 (공급 기반 크기 결정) | **대체형** (자격 공급자가 늘면 경매 지불액이 내려감 = 서로 대체) | 공급 기반 크기를 정량 동적 모형으로 다루지만 공급자는 동질 대체재이고 다품목 묶음·잉여배분·재참여 확률이 없다 | https://doi.org/10.1016/j.omega.2025.103328 | 초록(전문, SSRN 워킹페이퍼판) |
| 9 | A quantified Kraljic Portfolio Matrix: Using decision analysis for strategic purchasing | Montgomery, Ogden, Boehmke (2018) | J. of Purchasing and Supply Management (SSCI) | **의사결정분석(다속성 가치함수)** 로 KPM 두 축을 점수화 | 없음 | 확인 불가 (전문 미확인) | Kraljic을 **정량화**하지만 최적화·확률 모형이 아니라 **점수화·위치결정** 도구다 | https://doi.org/10.1016/j.pursup.2017.10.002 | 서지 + 피인용 42건 목록 |
| 10 | Positioning of commodities using the Kraljic Portfolio Matrix | Padhi, Wagner, Aggarwal (2012) | J. of Purchasing and Supply Management (SSCI) | KPM 위치결정 방법론 | 없음 | 확인 불가 (전문 미확인) | 같은 이유로 최적화 모형이 아니다 | https://doi.org/10.1016/j.pursup.2011.10.001 | 서지만 |
| 11 | A comprehensive framework and literature review of supplier selection under different purchasing strategies | Saputro, Figueira, Almada-Lobo (2022) | Computers & Industrial Engineering (SCIE Q1), 피인용 81 | 문헌 리뷰 + 통합 틀 (구매 전략별 공급자 선택 정량 모형 분류) | 없음 | 확인 불가 (전문 미확인) | **Kraljic 전략 유형과 정량 공급자선택 모형을 잇는 가장 유력한 교량 문헌**이지만 리뷰이고, 유지 인센티브·참여자 집합 동태는 범위 밖으로 보인다(전문 확인 필요) | https://doi.org/10.1016/j.cie.2022.108010 | 서지만 |
| 12 | Integrating supplier selection with inventory management under supply disruptions | Saputro, Figueira, Almada-Lobo (2021) | International J. of Production Research (SCIE Q1) | 공급 중단 하 공급자 선택 + 재고관리 통합 최적화 | 없음 | 확인 불가 (전문 미확인) | Kraljic 정량화 논문을 인용하는 최적화 모형이지만 공급자 이탈·유지 인센티브가 아니라 중단 위험이다 | https://doi.org/10.1080/00207543.2020.1866223 | 서지만 |
| 13 | Mitigating Supply Risk: Dual Sourcing or Process Improvement? | Wang, Gilland, Tomlin (2010) | M&SOM (SSCI, OM 주요) | 다수 공급자 조달 + **공급자 신뢰도 개선 노력(투자)** 동시 최적화, 확률적 용량·수율 | **공급자에 대한 개선 투자**(금전적 노력) | **대체형** (이중 소싱과 개선이 서로 **대체 수단**이며 공급자들은 같은 수요를 두고 경쟁) | 구매자가 공급자에게 자원을 쓰는 최적화 모형이지만 목적이 **유지(이탈 방지)가 아니라 신뢰도 개선**이고 정태적이다 | https://doi.org/10.1287/msom.1090.0279 (SSRN판 https://doi.org/10.2139/ssrn.4743858) | 초록(전문, SSRN판) |
| 14 | Disruption Risk and Optimal Sourcing in Multitier Supply Networks | Ang, Iancu, Swinney (2017) | Management Science (SSCI Q1) | 3계층 네트워크, 계약 파라미터로 tier-1의 소싱을 **간접 유도**; diamond-shaped 중복(overlap)이 최적 전략을 지배 | 간접 완화(계약·페널티) | **대체형(위험 상관 경로)** — 공유된 tier-2 공급자가 분산 효과를 무력화; 보완(동시 필요) 구조는 아님 | **공급 네트워크 구조가 최적 결정을 바꾼다**는 우리 실험 3의 취지와 통하지만 상호의존이 위험 상관을 통한 것이고 묶음 성립 조건이 아니다 | https://doi.org/10.1287/mnsc.2016.2471 | 초록(전문) |
| 15 | Compensating for Dynamic Supply Disruptions: Backup Flexibility Design | Saghafian & Van Oyen (2016) | Operations Research (SSCI Q1) | **다품목·다공급자**, 중단을 Markov chain으로, 재고부족 과정을 큐잉·댐 모형과 연결; "어떤 불신뢰 공급자를 백업할 것인가" | 백업 유연성 설계 | **대체형** (희소한 백업 유연성을 공급자들이 나눠 씀 = 경쟁) | 다품목·다공급자에서 **공급자별 중요도 순위**를 도출하는 동적 모형이라 우리 c_i와 목적이 가깝지만 순위 기준이 신뢰도·수요 분산의 2차 모먼트이고 **금전 유지 수단·묶음 all-or-nothing이 없다** | https://doi.org/10.1287/opre.2016.1478 (SSRN판 https://doi.org/10.2139/ssrn.2180355) | 초록(전문, SSRN판) |
| 16 | Managing Supply Disruptions when Sourcing from Reliable and Unreliable Suppliers | Hu & Kostamis (2015) | Production and Operations Management (SSCI Q1) | 근사 모형으로 다중 소싱 최적 정책; 불신뢰 주문의 **단순 순위 규칙**, 총량과 배분의 **독립성** | 없음 | **대체형** (공급자들이 같은 수요를 두고 경쟁, 순위화 가능) | 초록이 밝히는 "총 주문량과 배분이 **독립 결정**"은 우리 묶음 보완성과 정반대 구조(분리 가능성)다 | https://doi.org/10.1111/poms.12293 | 초록(전문) |
| 17 | Robust Sourcing Under Multilevel Supply Risks: Analysis of Random Yield and Capacity | Zhao, Freeman, Pan (2023) | INFORMS J. on Computing (SCIE Q1) | **분포적 강건 최적화**, 공급 위험의 **모호한 상관구조**, 다수준 중단 | 없음 | **대체형** (상관된 공급 위험을 두고 공급자 분산) | 공급자 간 상호의존을 상관구조로만 다루고 묶음 성립·유지 인센티브가 없다 | https://doi.org/10.1287/ijoc.2022.1254 | 초록(전문) |
| 18 | **Platform-Based Collaborative Routing using Dynamic Prices as Incentives** | Atasoy, Schulte, Steenkamp (2020) | Transportation Research Record (SCIE, 저널·프로시딩 혼합 성격) | **MIP** (협업 PDPTW): 플랫폼이 **캐리어에 지불하는 가격을 최소화**하면서 캐리어별 **개별합리성 제약**을 강제하는 동적 가격결정 | **동적 가격(지불액)** — 초록 명시: 플랫폼은 "물리적 자원을 직접 통제하지 못한다", 캐리어가 독자 운영보다 나아지게 보장 | **확인 불가** (협업 PDPTW에서 캐리어 간 관계가 요청 풀을 나누는 대체인지 경로 결합의 보완인지 초록으로 판정 불가) | **F1c에 가장 가까운 MIP 선례.** 그러나 인센티브가 **당기 개별합리성 제약**이고 **미래 재참여 확률·기간 간 전이·학습 VFA가 없다**; 다품목 all-or-nothing도 없다 | https://doi.org/10.1177/0361198120935116 | 초록(전문) |
| 19 | Profit allocation mechanisms for carrier collaboration in pickup and delivery service | Dai & Chen (2012) | Computers & Industrial Engineering (SCIE Q1) | 협력게임 이익배분 메커니즘(검색 스니펫 기준: 개별·집단 합리성, core) | 없음 (배분 규칙) | **보완형 추정** (캐리어 연합의 시너지 = 초가법성; **단, 초록 미확인**) | 배분 규칙이 **정태적 협력게임 해**이고 배분이 미래 참여를 바꾸는 동태가 없다 | https://doi.org/10.1016/j.cie.2011.11.029 | 서지 + 검색 스니펫 |
| 20 | The Interplay Between Supplier-Specific Investments and Supplier Dependence: Do Two Pluses Make a Minus? | Pulles, Ellegaard, Veldman (2023) | Journal of Management (SSCI Q1) | **실증**(두 개의 독립 dyad 데이터셋), 회귀·조절효과 | 관계 특수 투자 + 공급자 의존 → 공급자의 **자원 우선 배분** | **대체형** (공급자 자원을 두고 **구매자들이 경쟁**; 초록: relative to competing buyers, 의존이 투자 효과를 **음의 방향으로 조절**) | 관계 특수 투자가 공급자 자원 배분을 끌어온다는 실증 근거는 되지만 최적화 모형이 아니고 유지 확률이 아니라 자원 배분이다 | https://doi.org/10.1177/01492063221087643 | 초록(전문) |
| 21 | An Iterative Procurement Combinatorial Auction Mechanism for the Multi-Item, Multi-Sourcing Supplier-Selection and Order-Allocation Problem | Abbaas & Ventura (2024) | Mathematics (MDPI, SCIE — 사분위 개별 확인 안 함) | **반복 조달 조합경매 + MINLP**, 유연 입찰 언어, 가격민감 수요 | 없음 (가격 인하 압박) | **대체형 우세**(초록: several suppliers가 있는 경쟁 환경에서 가장 효과적, fostering competition and diversification); 유연 입찰 언어가 묶음 원가 보완을 담을 수 있으나 초록으로 확인 불가 → **혼재 가능** | 다품목·다공급자 조합 배정을 MINLP로 다루지만 **유지·재참여가 없고** 상호의존 활용 방향이 경쟁 촉진이다 | https://doi.org/10.3390/math12142228 | 초록(전문) |
| 22 | Dynamic Resource Allocation on Multi-Category Two-Sided Platforms | Li, Shen, Bart (2021) | Management Science (SSCI Q1) | 2범주·2기간 이론 모형, 범주 내 직접·간접 네트워크 효과 + **범주 간 상호의존**, 최적 자원배분 규칙(reinforcing 대 compensatory) | 플랫폼의 **투자 자원 배분**(양면·범주·기간에 걸쳐) | **혼재** (범주 간 상호의존 + 양면 네트워크 효과; 초록이 대체/보완 부호를 명시하지 않고 과금 모형에 따라 규칙이 뒤집힌다고 함) | 플랫폼 결정이 미래 성장(참여)을 바꾼다는 구조는 같으나 **집계 네트워크 효과 수준**이고 개별 참여자별 유지 가치·조합 매칭·묶음이 없다 | https://doi.org/10.1287/mnsc.2020.3586 | 초록(전문) |

---

## 2. F1a~F1d 직답

### F1a. Kraljic 계열 병목 품목·단일 공급원 위험을 **정량 최적화·확률 모형으로 조작화**한 연구

**직답: Kraljic 계보 안에서는 "정량화 = 점수화·위치결정(MCDM/의사결정분석)"에 그치고, 단일 공급원 위험을
확률·최적화 모형으로 다루는 작업은 Kraljic을 거의 인용하지 않는 별도 계보(공급 중단 문헌)에서 이루어진다.
두 계보를 잇는 문헌은 리뷰 1편(#11)이 유력 후보이며, 전문 확인이 필요하다.**

근거:
- Kraljic 정량화의 대표 문헌 #9(Montgomery, Ogden, Boehmke 2018 JPSM)는 제목·게재지 수준에서
  **decision analysis로 KPM을 정량화**한다고 밝힌다. #10(Padhi 외 2012 JPSM)도 positioning이다.
  두 편 모두 **확률적 최적화가 아니다**(전문 미확인이므로 그 이상은 주장하지 않는다).
- #9의 **피인용 42건을 Semantic Scholar로 확보해 전수 훑었다.** 그 안에서 최적화·확률 모형은
  #11(리뷰), #12(공급자 선택 + 재고, 공급 중단), Ai & Xu (2020) IJPR
  (https://doi.org/10.1080/00207543.2020.1711987, 서지만), Wu, Gao, Barnes (2022) IJPR
  (https://doi.org/10.1080/00207543.2022.2025945, 서지만) 정도이고, **나머지 대다수는 사례연구·AHP·DEA·설문**이었다.
  → **추론(직접 진술 아님)**: Kraljic 계보의 후속은 최적화보다 분류·점수화 쪽으로 발전했다.
- 단일 공급원 위험의 **정량 모형**은 별도 계보에 풍부하다: #5 Tomlin(2006)은 single sourcing을
  **최적 전략의 한 결과로 도출**하고, #4 Babich 외(2007)는 도산 상관과 경쟁을, #13 Wang 외(2010)는
  이중소싱 대 신뢰도 개선을, #14 Ang 외(2017)는 다계층 중복을, #15 Saghafian & Van Oyen(2016)은
  **다품목·다공급자에서 어떤 공급자를 백업할지**를, #17 Zhao 외(2023)는 모호 상관 하 강건 소싱을 다룬다.
  Yu, Zeng, Zhao (2009) Omega (https://doi.org/10.1016/j.omega.2008.05.006, 서지만)와
  Sawik (2014) COR (https://doi.org/10.1016/j.cor.2014.04.006, 서지만)도 같은 계보다.
- **공급 기반 크기 자체**를 동적 정량 모형으로 다룬 최신 문헌은 #8 Zhang & Katok(2025, Omega)이며,
  최적정지 + 이산 볼록 해석으로 control-limit/control-band 정책을 특성화한다.

**우리에게 주는 의미**: "병목 품목 = 공급자가 하나뿐인 품목"이라는 개념은 Kraljic 이래 표준이지만,
**그 병목성이 다품목 묶음의 성립 조건을 통해 다른 품목의 가치까지 결정한다**는 구조는 이 계보에서 찾지 못했다.
위 문헌들의 단일 공급원 위험은 모두 **그 품목 하나의 충족 위험**이다.

### F1b. 핵심 공급자 유지 인센티브를 최적화 모형으로 다룬 연구

**직답: 있다. 우리 아이디어 중 "금전으로 독립 공급자를 붙잡는다"는 부분은 이미 선례가 있다.
가장 강한 선례는 #1 Babich (2010) M&SOM이다. 이 점에서는 신규성을 주장할 수 없다.**

- **#1 Babich (2010)**: 동적·확률적 주기검토 모형에서 제조업체가 **용량예약과 금전 보조금을 동시에 결정**하고,
  최적 보조금 정책이 **subsidize-up-to** 구조를 가진다. 초록이 명시하듯 "제조업체의 행동, 예컨대 공급자에 대한
  금전 보조금이 공급자의 재무건전성에 깊이 영향을 준다"는 것이 출발점이다.
  → **"돈을 주어 공급자 생존 확률을 올리는 DP"의 정식 선례다.**
- **#2 Swinney & Netessine (2009)**: 구매자 매출이 공급자 도산을 좌우하는 2기간 계약 게임. 유지 수단은 **장기계약**.
- **#3 Tunca & Zhu (2018)**: **조기 지불·금융 중개**로 공급자 자금난 완화(게임 + 구조추정 + 반사실).
- **#13 Wang, Gilland, Tomlin (2010)**: 공급자에 대한 **개선 투자**의 최적 수준 (유지가 아니라 신뢰도 개선).
- **#20 Pulles 외 (2023, J. of Management)**: 실증. 구매자의 **관계 특수 투자**와 공급자 의존이 공급자의
  자원 배분을 끌어오지만, 의존이 투자 효과를 **음의 방향으로 조절**한다(두 플러스가 마이너스가 될 수 있다).
  → 우리 재참여 확률 함수의 단조 증가 가정에 대한 **반증 가능성**을 제기하는 실증 근거로 인용할 만하다.
- **물량 보장(volume guarantee) 자체**를 유지 수단으로 놓은 최적화 모형은 이번 검색으로 학술 문헌을 찾지 못했다.
  minimum volume commitment / take-or-pay 검색은 결과가 전부 실무 법률·구매 블로그였다(3.3절 쿼리 기록).

**남는 차이**: #1~#3·#13은 모두 (i) 구매자–공급자 dyad 또는 동질 대체 공급자 2명, (ii) 단일 품목,
(iii) 배분 몫이 아니라 보조금·계약 형태 결정, (iv) 학습된 가치함수 없음이다.
**묶음 all-or-nothing으로 인한 공급자 간 보완 상호의존은 어느 편에도 없다.**

### F1c. "소유하지 않은 공급자를 금전 인센티브로 유지하는 플랫폼·중개자" 모형 — 가장 중요

**직답: 유지(retention, 미래 재참여)까지 모형에 넣은 플랫폼·중개자 모형은 찾지 못했다.
찾은 것은 두 종류이며 둘 다 유지가 아니라 "당기 참여 유도"다.**

1) **#18 Atasoy, Schulte, Steenkamp (2020) TRR — 가장 가까운 MIP 선례.**
   초록이 우리 설정 문장과 거의 같은 말을 한다: "Uber, Uber Freight, Blackbuck, Lyft 같은 플랫폼 제공자는
   대부분 사람·재화를 움직이는 **물리적 자원을 직접 통제하지 못한다**", 따라서 "제3자가 협업하도록 **명확한
   인센티브를 설계하는 것이 결정적**"이고 "협업 인센티브는 플랫폼 제공자의 **운영 수준 의사결정 모형의
   필수 구성요소가 되어 동적으로 적용되어야 한다**". 실제 LTL 중개 플랫폼 사례에서 협업 PDPTW의
   **MIP를 세워 플랫폼이 캐리어에 지불하는 가격을 최소화하면서 캐리어별 개별합리성 제약을 강제**한다.
   결론도 "개별합리성을 강제해도 플랫폼 사업은 여전히 수익성이 있고, 캐리어에게 마진 증가를 보장할 수도 있다"이다.
   → **차이(결정적)**: 인센티브가 **당기 개별합리성 제약**이다. 이번 기간 지불이 **다음 기간 참여 확률**을 바꾸는
   기간 간 고리, 참여자 집합의 결정 의존 전이, 미래 가치 근사가 **없다**(초록 수준에서 그런 언급이 전혀 없다).
2) **#7 Martinez-de-Albeniz, Pinto, Amorim (2022) M&SOM — 가장 가까운 동적 선례.**
   플랫폼이 **수수료 구조**로 소유하지 않은 공급자의 참여를 유도하고, 공급자의 참여 결정을 재고·수요율·
   잔여기간의 함수로 **동적으로** 특성화한다. → **차이**: 공급자가 **1명**이라 공급자 간 상호의존이 없고,
   참여가 확률적 재참여가 아니라 수락 조건이며 조합 배정·묶음이 없다.
3) **#22 Li, Shen, Bart (2021) MS**: 플랫폼 자원배분이 미래 성장(참여)에 영향을 주는 동적 모형이지만
   **집계 네트워크 효과 수준**이고 개별 공급자 유지 가치가 없다.

**없다고 단정하는 부분** (지지 쿼리 목록은 3.3절):
- "플랫폼·중개자가 **개별 공급자에게 배분한 금액이 그 공급자의 다음 기간 재참여 확률을 정하고**,
  그 확률을 최적화 모형 안에 넣은" 모형은 **찾지 못했다.** 화물 중개, B2B 마켓플레이스, 조달 대행,
  계약 제조 중개, 4PL/5PL 모두에서 0건이다.
- **5PL 자체가 학술 정량 문헌으로 거의 존재하지 않는다**: fifth party logistics 5PL 검색에서 나온 학술
  결과는 Springer 단행본 장 2편뿐이었다(https://doi.org/10.1007/978-981-95-0533-3_6 ,
  https://doi.org/10.1007/978-981-95-0533-3_13 — 서지만 확인, 내용 미확인). 최적화 모형 논문은 0건.
- 마켓플레이스 판매자 유지 인센티브(티어별 커미션 할인, 로열티 할인)는 **실무에서 표준**이지만
  검색 결과가 전부 SEO 블로그였고 **학술 모형은 0건**이었다.

### F1d. 이 문헌들의 상호의존은 대체형인가 보완형인가 혼재인가

**직답: 압도적으로 대체형이다. 보완형은 협력적 운송 연합의 시너지(#19)와 플랫폼 범주 간 상호의존(#22)에서만
간접적으로 나타나고, 둘 다 "묶음 all-or-nothing 때문에 개별 공급자의 증분 가치가 독립 가치를 넘어선다"는
형태가 아니다.**

이유의 구조(**추론**임을 명시):
구매·공급위험 문헌의 표준 문제는 "**하나의 수요를 여러 공급자가 나눠 충족**"이다. 이 구조에서는 공급자를
늘리면 위험이 분산되지만 **수확 체감**이고, 공급자들은 같은 수요·같은 용량을 두고 경쟁하므로
V(S) − V(S∖{i}) ≤ V({i}) − V(∅) 방향(열성/대체형)이 자연스럽다. #4는 이 방향을 가장 명시적으로 보인다
(도산 상관이 낮을수록 경쟁이 약해져 구매자가 불리해짐). #16의 "총량과 배분이 독립 결정"은 **분리 가능성**이므로
보완성의 부재를 직접 진술하는 셈이다.
우리 설정에서 부호가 뒤집히는 이유는 **수요 자체가 묶음이고 all-or-nothing**이어서 다른 품목 공급자가
동시에 남아야 어떤 공급자의 기여도 실현되기 때문이다. 이 조건은 위 22편 중 **어느 편에도 없다.**

---

## 3. 상호의존 종류별 분류 표 (이번 조사의 핵심 산출물)

### 3.1 분류 요약

| 상호의존 종류 | 편수 | 문헌 # | 판정 근거 (초록 수준) |
|---|---|---|---|
| **대체형** (증분 < 독립: 용량·수요·자원을 두고 경쟁) | 10 | 2, 4, 5, 8, 13, 14, 15, 16, 17, 20 | 같은 수요를 나눠 충족 / 분산 대 최저가 상충 / 경매 경쟁 / 희소 백업 용량 공유 / 구매자들이 공급자 자원을 두고 경쟁 |
| **보완형** (증분 > 독립: 함께 있어야 가치가 생김) | 0 확정 · 1 추정 | (19 추정) | #19 캐리어 연합의 초가법적 시너지는 보완형으로 보이지만 **초록 미확인**이라 확정하지 않는다 |
| **혼재** | 2 | 21(가능), 22 | #21은 경쟁 강조 + 유연 입찰 언어의 묶음 원가 보완 가능성 / #22는 범주 간 상호의존 + 양면 네트워크 효과, 부호가 과금 모형에 따라 뒤집힘 |
| **해당 없음** (개체 1명 또는 상호의존 미정의) | 4 | 1, 3, 6, 7 | 공급자 1명 dyad 또는 유형 2개뿐 |
| **확인 불가** (전문 필요) | 5 | 9, 10, 11, 12, 18 | 위치결정·리뷰이거나 초록만으로 캐리어 간 관계를 판정할 수 없음 |

### 3.2 G1 현상 문장에 대한 판정

ref/literature-review.md의 새 G1: **"다품목 묶음 주문이 만드는 보완형 상호의존 때문에 참여자의 증분 가치가
독립 가치를 넘어서며, 이는 용량 경쟁의 대체형(VIC < CLV)과 부호가 반대다."**

F1 범위(구매 관리·공급망 위험)에서 이 문장을 **약화하는 증거는 찾지 못했다.** 오히려 지지한다:
- 이 계보의 표준 상호의존은 대체형이고(3.1), 부호가 반대라는 진술 자체는 #4·#16처럼 명시적으로 확인된다.
- 다만 **F1은 G1을 "새롭다"고 증명하지 못한다.** 왜냐하면
  (a) F1b의 #1 Babich(2010)이 "금전으로 독립 공급자를 유지하는 동적 모형"을 이미 갖고 있고,
  (b) 보완형 상호의존 자체는 조합경매·협력게임(#19, #21)에서 익숙한 개념이므로,
  남는 신규성은 **"금전 유지 + 묶음 보완 + 개별 참여자별 미래가치 계수"의 조합**에 있다.
  이 조합을 F2(네트워크 RM)·F3(신뢰도 중요도)·F4(선행 확인) 결과와 합쳐 판단해야 한다.
- **반박 가능성(누가 "이미 있다"고 말할 수 있는가)**: 리뷰어가 #1 Babich(2010)을 들어
  "구매자가 돈으로 공급자 생존 확률을 조절하는 동적 모형은 2010년에 이미 있다"고 지적할 수 있다.
  우리 답변은 "그 모형은 공급자 1명·단일 품목이라 **c_i(증분 가치)라는 양 자체가 정의되지 않는다**"여야 한다.

### 3.3 확인한 쿼리 목록 (결과 없음 또는 SEO만 — F1c 단정의 근거)

학술 결과 0건이었던 쿼리(모두 실제 실행함):

| 쿼리 | 결과 |
|---|---|
| platform pays incentive to retain independent sellers marketplace dynamic programming seller churn commission | 상위 9건 전부 SEO 블로그(wcvendors, cs-cart, marketplacer, flxpoint, origami-marketplace, journeyh.io, dealhub, hoangtrungdigital) + SEC 공시 1건. 학술 0건 |
| "contract manufacturing" OR "procurement outsourcing" intermediary retains suppliers monetary incentive optimization model | McKinsey 인사이트 2건, PRGX 가이드, 미국 특허 2건, Wikipedia. 학술 최적화 모형 0건 |
| minimum volume commitment quantity guarantee keep supplier capacity dynamic program buyer pays premium survival | 전부 실무·법률 페이지(xeneta, solvimon, lawinsider, jdsupra, cello-square 등). 학술 0건 |
| procurement intermediary agent supplier loyalty payment model B2B marketplace matching surplus sharing | 전부 B2B 소프트웨어 마케팅 페이지. 학술 0건 |
| fourth party logistics 4PL platform profit allocation carriers participation incentive optimization model | 4PL 소개 블로그 8건 + 3PL 역경매 메커니즘 설계 1건(PMC6261608, 유지 아님). 유지 모형 0건 |
| "fifth party logistics" 5PL platform optimization model academic matching | Springer 단행본 장 2편(서지만), 나머지 SEO. **5PL 정량 최적화 논문 0건** |
| "revenue sharing" platform intermediary "supplier participation" dynamic program retention probability operations research | Balseiro 외 Dynamic Revenue Sharing(광고 거래소, 유지 아님), Cachon & Lariviere MS 2005(수익공유 계약, 유지 아님), 나머지 SEO |
| "purchasing portfolio" Kraljic stochastic programming sourcing strategy optimization model bottleneck strategic items | ResearchGate 사례연구 3건 + SEO. Kraljic를 확률계획으로 조작화한 논문 0건 |
| multi-item all-or-nothing order supplier participation retention platform optimization complementary suppliers bundle procurement | 학술 2건(#21, 그리고 two-tier 주문배정 arXiv 2210.11953) + 미국 특허 6건. **유지·재참여를 다룬 것 0건** |
| site:arxiv.org platform surplus sharing supplier participation probability dynamic matching retention MDP | 매칭·메커니즘 논문 다수이나 **배분액이 미래 참여를 바꾸는 것 0건** |

추가로 시도했으나 우리 조건(배분 → 미래 재참여)을 만족하는 결과가 없던 각도:
supplier exit / supplier attrition endogenous buyer decision dynamic model;
site:pubsonline.informs.org supply base reduction single sourcing risk stochastic model;
site:sciencedirect.com freight forwarder platform carrier participation incentive surplus allocation MIP individual rationality;
freight brokerage carrier retention incentive dynamic model platform;
digital freight platform carrier retention dynamic matching reinforcement learning;
supplier development investment optimization model keep supplier from exiting market dynamic program.

---

## 4. 미해결 · 확인 불가 (페이월 목록 + 확인할 질문)

전문을 한 편도 읽지 못했다(sciencedirect·informs 모두 403). 도서관 접속으로 확인할 구체적 질문:

| # | 문헌 | 확인할 질문 |
|---|------|-------------|
| 18 | Atasoy 외 (2020) TRR | (a) 인센티브가 **기간 간(다음 기간 참여)** 효과를 갖는가, 아니면 당기 IR 제약뿐인가? (b) 캐리어 간 관계가 요청 풀 경쟁(대체)인가 경로 결합 시너지(보완)인가? (c) MIP에 미래 가치 항이 있는가? |
| 7 | Martinez-de-Albeniz 외 (2022) M&SOM | 공급자가 **복수**인 확장이 있는가? 있으면 공급자 간 관계가 대체(재고 경쟁)인가? 참여가 **확률적**으로 모형화된 절이 있는가? |
| 1 | Babich (2010) M&SOM | 보조금에서 공급자 생존 확률로 가는 **함수 형태**는 무엇인가(로지스틱인가, firm-value 임계값인가)? **다수 공급자 확장**에서 보조금의 상호의존 부호를 논의하는가? |
| 11 | Saputro 외 (2022) C&IE 리뷰 | Kraljic의 **병목·전략 품목 구분을 최적화 모형의 구조로 번역한** 논문이 이 리뷰 안에 있는가? 있으면 어느 편인가? |
| 9 | Montgomery 외 (2018) JPSM | 정량화가 **점수화에 그치는지**, 아니면 그 점수를 최적화 목적함수에 넣는 절이 있는지 |
| 15 | Saghafian & Van Oyen (2016) OR | "어떤 공급자를 백업할지"의 중요도 지표가 **leave-one-out 차분 형태**인가? 다품목인데 품목 간 **동시 필요(묶음)** 조건이 있는가? |
| 19 | Dai & Chen (2012) C&IE | 이익배분이 **core**인지, 연합 가치가 **초가법적**인지(보완형 확정에 필요). 초록이 Crossref에 없어 확인 못 함 |
| 20 | Pulles 외 (2023) JOM | 의존의 **음의 조절효과**가 "많이 주면 오히려 덜 붙잡힌다"를 뜻하는가? 우리 로지스틱 단조성 가정에 대한 반증으로 인용 가능한가? |
| — | Ai & Xu (2020) IJPR; Wu 외 (2022) IJPR; Yu 외 (2009) Omega; Sawik (2014) COR | 서지만 확인했다. 단일/이중 소싱 결정에서 상호의존이 대체형뿐인지 확인 필요 |
| — | OpenAlex Kraljic 1983 피인용 전수 | **재시도 필요.** API 예산이 UTC 자정에 리셋되므로 다음 세션에 api.openalex.org 로 Kraljic 1983 id를 찾아 filter=cites: 로 전수 훑고, "Kraljic 계보에는 확률 최적화가 드물다"는 추론을 검증해야 한다 |

### 과제별 상태

| 과제 | 상태 |
|---|---|
| F1a | **done** (단, Kraljic 1983 직접 피인용 전수는 OpenAlex 불가로 미완 → needs follow-up) |
| F1b | **done** (핵심 선례 #1 Babich 2010 확정) |
| F1c | **done** (없다고 단정 + 쿼리 목록 기록; 가장 가까운 것은 #18, #7) |
| F1d | **done** (대체형 압도 — 3.1 표) |
| Kraljic 1983 OpenAlex 피인용 순회 | **blocked** (OpenAlex 일일 예산 소진; 다음 세션 재시도) |
| 전문(full-text) 확인 | **blocked** (출판사 403; 4절 질문 목록) |

---

## 5. 출처 목록

1. Babich, V. (2010) Independence of Capacity Ordering and Financial Subsidies to Risky Suppliers. M&SOM. https://doi.org/10.1287/msom.1090.0284
2. Swinney, R. & Netessine, S. (2009) Long-Term Contracts Under the Threat of Supplier Default. M&SOM. https://doi.org/10.1287/msom.1070.0199
3. Tunca, T.I. & Zhu, W. (2018) Buyer Intermediation in Supplier Finance. Management Science. https://doi.org/10.1287/mnsc.2017.2863
4. Babich, V., Burnetas, A.N. & Ritchken, P.H. (2007) Competition and Diversification Effects in Supply Chains with Supplier Default Risk. M&SOM. https://doi.org/10.1287/msom.1060.0122
5. Tomlin, B. (2006) On the Value of Mitigation and Contingency Strategies for Managing Supply Chain Disruption Risks. Management Science. https://doi.org/10.1287/mnsc.1060.0515
6. Yang, Z.B., Aydin, G., Babich, V. & Beil, D.R. (2009) Supply Disruptions, Asymmetric Information, and a Backup Production Option. Management Science. https://doi.org/10.1287/mnsc.1080.0943
7. Martinez-de-Albeniz, V., Pinto, C. & Amorim, P. (2022) Driving Supply to Marketplaces: Optimal Platform Pricing When Suppliers Share Inventory. M&SOM. https://doi.org/10.1287/msom.2022.1105
8. Zhang, W. & Katok, E. (2025) To stop or not, that is the question: When should a buyer hit the brakes for supply base rationalization? Omega. https://doi.org/10.1016/j.omega.2025.103328 (워킹페이퍼판 https://doi.org/10.2139/ssrn.4921029)
9. Montgomery, R.T., Ogden, J.A. & Boehmke, B.C. (2018) A quantified Kraljic Portfolio Matrix. J. of Purchasing and Supply Management. https://doi.org/10.1016/j.pursup.2017.10.002
10. Padhi, S.S., Wagner, S.M. & Aggarwal, V. (2012) Positioning of commodities using the Kraljic Portfolio Matrix. JPSM. https://doi.org/10.1016/j.pursup.2011.10.001
11. Saputro, T.E., Figueira, G. & Almada-Lobo, B. (2022) A comprehensive framework and literature review of supplier selection under different purchasing strategies. Computers & Industrial Engineering. https://doi.org/10.1016/j.cie.2022.108010
12. Saputro, T.E., Figueira, G. & Almada-Lobo, B. (2021) Integrating supplier selection with inventory management under supply disruptions. IJPR. https://doi.org/10.1080/00207543.2020.1866223
13. Wang, Y., Gilland, W. & Tomlin, B. (2010) Mitigating Supply Risk: Dual Sourcing or Process Improvement? M&SOM. https://doi.org/10.1287/msom.1090.0279 (SSRN판 https://doi.org/10.2139/ssrn.4743858)
14. Ang, E., Iancu, D.A. & Swinney, R. (2017) Disruption Risk and Optimal Sourcing in Multitier Supply Networks. Management Science. https://doi.org/10.1287/mnsc.2016.2471
15. Saghafian, S. & Van Oyen, M.P. (2016) Compensating for Dynamic Supply Disruptions: Backup Flexibility Design. Operations Research. https://doi.org/10.1287/opre.2016.1478 (SSRN판 https://doi.org/10.2139/ssrn.2180355)
16. Hu, B. & Kostamis, D. (2015) Managing Supply Disruptions when Sourcing from Reliable and Unreliable Suppliers. POM. https://doi.org/10.1111/poms.12293
17. Zhao, M., Freeman, N.K. & Pan, K. (2023) Robust Sourcing Under Multilevel Supply Risks. INFORMS J. on Computing. https://doi.org/10.1287/ijoc.2022.1254
18. Atasoy, B., Schulte, F. & Steenkamp, A. (2020) Platform-Based Collaborative Routing using Dynamic Prices as Incentives. Transportation Research Record. https://doi.org/10.1177/0361198120935116
19. Dai, B. & Chen, H. (2012) Profit allocation mechanisms for carrier collaboration in pickup and delivery service. Computers & Industrial Engineering. https://doi.org/10.1016/j.cie.2011.11.029
20. Pulles, N.J., Ellegaard, C. & Veldman, J. (2023) The Interplay Between Supplier-Specific Investments and Supplier Dependence. Journal of Management. https://doi.org/10.1177/01492063221087643
21. Abbaas, O. & Ventura, J.A. (2024) An Iterative Procurement Combinatorial Auction Mechanism for the Multi-Item, Multi-Sourcing Supplier-Selection and Order-Allocation Problem. Mathematics. https://doi.org/10.3390/math12142228
22. Li, H., Shen, Q. & Bart, Y. (2021) Dynamic Resource Allocation on Multi-Category Two-Sided Platforms. Management Science. https://doi.org/10.1287/mnsc.2020.3586

보조(서지만 확인, 초록·본문 미확인):

23. Ai, Y. & Xu, Y. (2020) Strategic sourcing in forward and spot markets with reliable and unreliable suppliers. IJPR. https://doi.org/10.1080/00207543.2020.1711987
24. Wu, C., Gao, J. & Barnes, D. (2022) Sustainable partner selection and order allocation for strategic items. IJPR. https://doi.org/10.1080/00207543.2022.2025945
25. Yu, H., Zeng, A.Z. & Zhao, L. (2009) Single or dual sourcing: decision-making in the presence of supply chain disruption risks. Omega. https://doi.org/10.1016/j.omega.2008.05.006
26. Sawik, T. (2014) Optimization of cost and service level in the presence of supply chain disruption risks: Single vs. multiple sourcing. COR. https://doi.org/10.1016/j.cor.2014.04.006
27. Pfeiffer, T. (2010) A dynamic model of supplier switching. EJOR. https://doi.org/10.1016/j.ejor.2010.05.030
28. Deng, Y. 외 (2022) Incentive design and profit sharing in multi-modal transportation networks. Transportation Research Part B. https://doi.org/10.1016/j.trb.2022.06.011
29. van Heeswijk, W. (2022) Strategic bidding in freight transport using deep reinforcement learning. Annals of OR. https://doi.org/10.1007/s10479-022-04572-z
30. Fifth-Party Logistics (5PL), Springer 단행본 장. https://doi.org/10.1007/978-981-95-0533-3_6 · Fifth-Party Logistics (5PL): Mastering Digital Supply Chain Ecosystems. https://doi.org/10.1007/978-981-95-0533-3_13

---

## 6. Coverage Status

**직접 확인한 것**
- Crossref/SSRN 등록 초록 **전문**을 읽은 논문 14편 (표에서 "초록(전문)" 표기분: #1~#8, #13~#17, #20~#22).
- Kraljic 정량화 논문(#9)의 **피인용 42건 목록 전수** (Semantic Scholar; 제목·연도·게재지·피인용수 수준).
- F1c 각도 **10개 쿼리의 결과 전수** (3.3절) — 학술 결과 0건임을 확인.

**여전히 불확실한 것**
- #18 Atasoy 외의 캐리어 간 상호의존 종류(대체/보완)와 기간 간 효과 유무 → 전문 필요.
- "Kraljic 계보에 확률 최적화가 드물다"는 판단은 #9의 피인용 42건 + 검색에 근거한 **추론**이다.
  Kraljic 1983 원문의 직접 피인용 전수(수천 건)는 확인하지 못했다.
- #19의 보완형 판정은 **검색 스니펫 수준**이라 확정하지 않았다.
- 저널 사분위는 개별 확인하지 않았다(통상적 평판 표기).

**완료하지 못한 작업**
- **OpenAlex 인용 그래프 순회** — 지시받은 주 수단이나 API 일일 예산 소진으로 불가. 다음 세션 재시도 필요.
- **전문(full-text) 읽기** — 출판사 403으로 0편. 4절 질문 목록으로 넘긴다.
