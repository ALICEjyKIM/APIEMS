# 주제 A 증거 수집: 내생적 참여자 구성 (플랫폼 결정 → 미래 참여자 집합)

작성일: 2026-09-28
수집 도구: WebSearch(각도별 병렬 10여 회), OpenAlex API(제목/초록 검색 + 참조·피인용 순회), WebFetch(arXiv/ar5iv/NeurIPS PDF)
목적: 문헌 요약이 아니라 **갭 특정**. `ref/literature-review.md`의 A1–A5에 답한다.
기준 설정: 5PL 미들마일 B2B 플랫폼, 매 기간 (주문 수락 + 품목별 공급자·수량 배정 + 3자 잉여배분)을 **하나의 MILP**로 결정,
all-or-nothing 다품목 묶음, 재참여확률 = 로지스틱(배분 잉여율), 공급자 슬롯 고정·빈자리 확률 0.25 충원.

**질문 진행 상태**: A1 done / A2 done / A3 done(강건성 문헌은 negative) / A4 done(negative 확정) / A5 done

---

## 1. 증거 표

읽은 수준: `전문`=본문 해당 절을 직접 추출해 읽음, `초록`=초록 전문을 직접 확보, `서지`=서지정보만 확인(내용 추정 금지).
게재지 등급: **사분위는 직접 확인하지 않았다.** 저널명과 본 프로젝트 기준(OR/OM 주요 저널 여부)만 적는다.

| # | 제목 | 저자·연도 | 게재지(등급) | 결정변수 | 참여자 집합 내생 | 반응 함수 | 벤치마크 | 우리 설정과의 차이 | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | On-Demand Service Platforms | Taylor 2018 | M&SOM (사분위 미확인) | 서비스당 가격·임금(정태) | 참여 여부가 가격·임금에 내생(단일 기간 균형) | 고객 가치·에이전트 기회비용 불확실성 하 참여 임계(유보값) | 불확실성 없는 기준 설정 | 단일 기간 균형. 기간 간 참여자 집합 전이·매칭·다품목 없음 | https://doi.org/10.1287/msom.2017.0678 | 초록 |
| 2 | The Role of Surge Pricing on a Service Platform with Self-Scheduling Capacity | Cachon, Daniels, Lobel 2017 | M&SOM (사분위 미확인) | 가격·임금 계약 형태(고정/서지) | 자기일정 공급자 참여가 보수에 내생(정태) | 유보임금 임계 | 고정가격 계약 등 계약군 | 정태 균형. 상태 전이 없음 | https://doi.org/10.1287/msom.2017.0618 | 초록 |
| 3 | Your Uber Is Arriving: Managing On-Demand Workers... | Guda & Subramanian 2019 | Management Science (사분위 미확인) | 서지가격·예측정보 공개·인센티브 | 노동자 이동/참여 내생(게임) | 전략적 노동자 최적반응 | 정보 비공개/서지 없음 | 정태 게임. 잉여배분 몫을 기간별 결정변수로 두지 않음 | https://doi.org/10.1287/mnsc.2018.3050 | 초록 |
| 4 | Surge Pricing and Its Spatial Supply Response | Besbes, Castro, Lobel 2021 | Management Science (사분위 미확인) | 공간별 가격 | 공급 재배치가 가격에 내생 | 공급자 공간 반응 | 균일가격 | 공급 "위치" 반응이며 참여자 집합의 소멸·충원이 아님 | https://doi.org/10.1287/mnsc.2020.3622 | 초록 |
| 5 | Labor Welfare in On-Demand Service Platforms | Benjaafar, Ding, Kong, Taylor 2022 | M&SOM (사분위 미확인) | 가격·임금 | 노동 참여 내생(정태) | 기회비용 분포 기반 참여 | 임금 규제 등 정책 | 후생 분석. 동적 매칭·잉여배분 결정 없음 | https://doi.org/10.1287/msom.2020.0964 | 초록 |
| 6 | Dynamic Type Matching | Hu & Zhou 2022 | M&SOM (사분위 미확인) | 기간별 유형 간 매칭 수량 | 아니오(미매칭분 이월, 대기·보유비용) | 없음 | 최적정책 구조(우선순위·match-down-to), 휴리스틱 | 매칭만 결정. 잉여배분·이탈확률 없음. 묶음 아님 | https://doi.org/10.1287/msom.2020.0952 | 초록 |
| 7 | Dynamic Stochastic Matching Under Limited Time | Aouad & Sarıtaç 2022 | Operations Research (사분위 미확인) | 누구를·언제 매칭할지 | 부분적(매칭되면 이탈, 미매칭은 외생 확률로 포기) | 외생·이질적 abandonment rate | LP 벤치마크(상한), 배치 알고리즘, 상수비 근사 보증 | 이탈확률이 **배분 잉여에 의존하지 않음**. 금전 배분 결정 없음 | https://doi.org/10.1287/opre.2022.2293 | 초록 |
| 8 | Ride-Hailing Networks with Strategic Drivers | Afèche, Liu, Maglaras 2023 | M&SOM (사분위 미확인) | 입장통제·가격·재배치 | 기사 재배치·진입 내생(전략적) | 전략적 기사 균형 | 통제능력 수준별 비교 | 기간별 잉여배분 결정 없음 | https://doi.org/10.1287/msom.2023.1221 | 서지 |
| 9 | Balancing Agent Retention and Waiting Time in Service Platforms | Musalem, Olivares, Yung 2023 | Operations Research 71(3):979–1003 (사분위 미확인) | 프리랜스 에이전트 용량(스태핑) 수준 | 예. 낮은 가동률·스케줄 변동성 → 자발적 이직 증가 | 데이터로 추정한 이직 모형 | 데이터 기반 용량정책 비교(실제 콜센터) | 결정변수가 **용량**. 잉여배분·매칭·묶음 없음 | https://doi.org/10.1287/opre.2022.2418 | 초록 |
| 10 | Strategic Workforce Planning in Crowdsourced Delivery With Hybrid Driver Fleets | Luy, Hiermann, Schiffer 2024 | Production and Operations Management (사분위 미확인) | 기간별 고정기사(FD) 채용 수 | **예.** 크라우드 기사 이탈확률이 미매칭 비율의 함수 | p = p_high·(미매칭 비율) + p_low·(1−미매칭 비율), 이탈 수는 이항분포 | 근시안(MY) 채용정책; 완전정보 lookahead(초록) | 결정변수가 채용 수. VFA가 FD 차원의 **분리형 구간선형**. 잉여배분·묶음·그래프 없음 | https://doi.org/10.1177/10591478241268602 · https://arxiv.org/abs/2311.17935 | 전문(모델·실험 절) |
| 11 | An Approximate Dynamic Programming Approach to Dynamic Stochastic Matching | You & Vossen 2024 | INFORMS Journal on Computing (사분위 미확인) | 기간별 매칭 | 아니오(도착·이탈 외생) | 없음 | **ALP 기반 상·하한과 최적성 격차**, 실패 가능 매칭 | 참여자 집합이 결정에 의존하지 않음. 금전 배분 없음 | https://doi.org/10.1287/ijoc.2021.0203 | 초록 |
| 12 | Dynamic optimization strategies for on-demand ride services platform: Surge pricing, commission rate, and incentives | Chen, Zheng, Ke, Yang 2020 | Transportation Research Part B (사분위 미확인) | **기간별 서지가격 + 수수료율 + 인센티브** | 예. 전략이 승객·공차 도착률에 영향(상태의존 도착률) | 동적 vacant car–passenger meeting 모형(도착률 반응) | 근시안 대비 non-myopic ADP | 수수료율이 **단일 스칼라**(참여자별 배분 아님), 개별 재참여확률 아님, 묶음·MILP 없음 | https://doi.org/10.1016/j.trb.2020.05.005 | 초록(검색 요약 경유, medium) |
| 13 | Dynamic Revenue Sharing | Balseiro, Lin, Mirrokni, Paes Leme, Zuo 2017 | NeurIPS 2017 (학회) | **기간별 수익배분 몫 α** | 아니오(참여는 IR 제약으로 강제) | 없음. 판매자 기회비용 c 이상 지급이 **기대값 제약** | 정적 예비가격 메커니즘, 기존 정적 수익배분 | 참여가 하드 IR 제약. 참여자 집합 불변 | https://papers.nips.cc/paper/2017/file/cb8acb1dc9821bf74e6ca9068032d623-Paper.pdf | 전문(요약 추출) |
| 14 | Fending Off Critics of Platform Power with Differential Revenue Sharing | Bhargava, Wang, Zhang 2022 | Management Science (사분위 미확인) | 생산자 규모별 차등 수익배분율 | 예(참여·산출이 배분율에 내생, 정태) | 생산자 참여·산출 최적반응 | 단일 선형 수익배분 설계 | 정태 설계. 기간 간 전이·동적계획 없음 | https://doi.org/10.1287/mnsc.2022.4545 | 초록 |
| 15 | Supply Chain Coordination with Revenue-Sharing Contracts | Cachon & Lariviere 2005 | Management Science (사분위 미확인) | 도매가 + 수익배분 비율 | 아니오 | 없음 | 도매가 계약 등 | 정태 계약이론. 이익 임의 배분 가능성은 보이나 참여자 동태 없음 | https://doi.org/10.1287/mnsc.1040.0215 | 초록 |
| 16 | Optimizing last-mile delivery: a dynamic compensation strategy for occasional drivers | Schur & Winheller 2024 | OR Spectrum (사분위 미확인) | 도착 기사별 **개별 보상액 + 과업 묶음** | 아니오(1회 수락 여부만) | 수락 임계값을 확률변수로 | 고정보상 정책, 근사법 간 비교 | 기간 간 재참여 없음. 개별 보상 결정 자체는 우리와 유사 | https://doi.org/10.1007/s00291-024-00796-6 | 초록 |
| 17 | Fairness as an Investment: Dynamic Participation and Long-Run Profit in Virtual Power Plants | Chen & Xu 2026 (프리프린트, 미심사) | arXiv | 소비자별 **배분량 D_{i,t}** | **예.** S_{i,t+1}=βS_{i,t}+ρD_{i,t}, 가용성 A=g(S) | g(x)=1−exp(−ηx), 증가·오목·포화 | 근시안(greedy 이익최대), 엄격 공정성, slack-augmented | 배분 대상이 **수량**(금전 잉여 아님), 결정론적 DP, 학습 VFA·묶음·그래프 없음 | https://arxiv.org/abs/2606.02820 | 전문(모델·벤치마크 절) |
| 18 | To Start Up a Start-Up — Embedding Strategic Demand Development ... via RL with Information Shaping | Chen, Ulmer, Thomas 2025 (프리프린트) | arXiv | 지역별 차량 배분 | **예.** 지역 서비스품질 → 그 지역 수요 성장 | 초록에 함수형 명시 없음(확인 불가) | 초록에 명시 없음 | 수요(주문자) 측 성장만. 공급자 이탈·잉여배분 없음 | https://arxiv.org/abs/2504.05633 | 초록 |
| 19 | Threshold-based incentives for ride-sourcing drivers | Liu, Xu, Vignon, Yin, Qin, Li 2023 | Transportation Research Part C (사분위 미확인) | 임계 기반 인센티브 설계 | 확인 필요 | 제목상 **임계형** | 확인 필요 | 초록 미확보. 내용 단정 금지 | https://doi.org/10.1016/j.trc.2023.104323 | 서지 |
| 20 | Thickness and Information in Dynamic Matching Markets | Akbarpour, Li, Oveis Gharan 2020 | Journal of Political Economy (사분위 미확인) | 언제 매칭할지(대기 vs greedy) | 부분적(미매칭 에이전트가 외생 확률로 소멸) | 외생 소멸률 | greedy vs patient 정책, 이론 보증 | 소멸이 금전 배분과 무관 | https://doi.org/10.1086/704761 | 초록 |
| 21 | Optimal Hiring and Retention Policies for Heterogeneous Workers Who Learn | Arlotto, Chick, Gans 2014 | Management Science (사분위 미확인) | 채용·유지 결정 | 예(인력 구성이 결정에 의존) | 학습·이직 모형 | 최적정책 구조 | 금전 배분·매칭·묶음 없음. 인력계획 계보 원전 | https://doi.org/10.1287/mnsc.2013.1754 | 서지 |
| 22 | Managing Learning and Turnover in Employee Staffing | Gans & Zhou 2002 | Operations Research (사분위 미확인) | 채용 수 | 예(이직률 하 인력 구성) | 이직률(외생 파라미터) | 최적정책 구조 | 동일 | https://doi.org/10.1287/opre.50.6.991.343 | 서지 |
| 23 | Decision-dependent probabilities in stochastic programs with recourse | Hellemo, Barton, Tomasgård 2018 | Computational Management Science (사분위 미확인) | (방법론) | 결정의존 확률의 일반 정식화 | — | — | 우리 문제의 "결정의존 불확실성" 라벨을 주는 방법론 원전 | https://doi.org/10.1007/s10287-018-0330-0 | 서지 |
| 24 | Approximate Dynamic Programming (2nd ed.) | Powell 2011 | Wiley 단행본 | (방법론) | — | — | 근시안·후견(완전정보) 상한 등 ADP 표준 벤치마크 어휘 | 우리 VFA·벤치마크 어휘의 원전 | https://doi.org/10.1002/9781118029176 | 서지 |
| 25 | Frailty-Aware Transformer for Recurrent Survival Modeling of Driver Retention | Xu, Zhang, Miller 2025 (프리프린트) | arXiv | (예측) | — | **학습된 반복 생존 모형** | 초록 미확보 | 반응함수를 학습하는 최신 사례. 최적화와 결합 안 됨 | https://arxiv.org/abs/2511.19893 | 서지 |
| 26 | There Is More to Crowdshipping Than Money | Masorgo, Dobrzykowski, Tang, Fugate 2026 | Journal of Operations Management (사분위 미확인) | (실증) | — | 금전 외 운영 특성이 기사 반응에 영향 | — | 반응함수가 잉여율 단일 변수가 아닐 수 있다는 실증 반박 근거 | https://doi.org/10.1002/joom.70046 | 서지 |
| 27 | Dynamic Pooling and Regional Participation in Deceased-Donor Organ Allocation | Okada 2026 (프리프린트) | arXiv | 우선순위 지수(배분 규칙) | 예(지역 참여 제약) | 참여제약(IC/IR), 확률적 재참여 아님 | 분권 vs 통합, 공리주의 배분 | 참여가 하드 제약. 금전 배분 아님 | https://arxiv.org/abs/2609.18147 | 초록 |
| 28 | On-Demand Service Sharing via Collective Dynamic Pricing | Dogan & Jacquillat 2025 | M&SOM (사분위 미확인) | 기간별 서비스 배분 + 가격(메커니즘) | 아니오(IC·IR 제약) | 없음 | 고정가(posted price) | DP로 분해하지만 참여자 집합 전이 없음 | https://doi.org/10.1287/msom.2024.1301 | 초록 |
| 29 | Dynamic Workforce Acquisition for Crowdsourced Last-Mile Delivery Platforms | Lei, Jasin, Wang, Deng 2020 (워킹페이퍼) | SSRN | 기간별 인력 확보 | 확인 필요 | 확인 필요 | 확인 필요 | 초록 미확보 | https://doi.org/10.2139/ssrn.3532844 | 서지 |
| 30 | Spatial Pricing in Ride-Sharing Networks | Bimpikis, Candogan, Sabán (연도 미확정: OpenAlex 2017) | Operations Research (사분위 미확인) | 지역별 가격 | 예(공급 이동 내생, 균형) | 균형 공급 반응 | 균일가격 | 정태 네트워크 균형 | https://openalex.org/W2766210608 | 서지 |
| 31 | Learning Pay Strategies with Small Samples in Gig Economy Platforms | Delarue, Lian, Qin 2026 (워킹페이퍼) | SSRN | 보수(pay) 전략 | 확인 필요 | 확인 필요 | 확인 필요 | 403으로 초록 확보 실패. A2 답을 바꿀 수 있는 후보 | https://doi.org/10.2139/ssrn.6197638 | 서지 |
| 32 | Determinants of fifth party logistics (5PL) | Hosie, Sundarakani, Tan 2012 | Int. J. of Logistics Systems and Management (사분위 미확인) | (개념·실증) | 아니오 | 없음 | 없음 | 5PL 문헌은 개념·역량 논의. 최적화 모형 아님 | https://doi.org/10.1504/ijlsm.2012.049700 | 서지 |
| 33 | Digital transformation in ecosystems: ... application to fifth-party logistics | Nicoletti & Appolloni 2024 | J. of Global Operations and Strategic Sourcing (사분위 미확인) | (개념 프레임워크) | 아니오 | 없음 | 없음 | 동일 | https://doi.org/10.1108/jgoss-04-2023-0024 | 서지 |
| 34 | Not All Matches Are Equally Valuable: Retention-Focused Recommendation in a Job-Matching Platform | (저자 미확인) 2026 (프리프린트) | arXiv | 추천 랭킹 보정 | 예(추천 → 이탈) | 최근 매칭 수에 대한 **이탈위험 임계**(함수형 미공개) | 대조군(온라인 실험) | 실험 연구. 최적화 모형 아님 | https://arxiv.org/abs/2609.01652 | 초록 |

---

## 2. 가장 가까운 선행연구 4편과 차이

### (1) Luy, Hiermann, Schiffer (2024), *Production and Operations Management* — 구조적으로 가장 가까움 [10]
- 공통점: 다기간 MDP + **ADP로 미래가치 근사** + **참여자 이탈이 당기 결정에 내생**. 크라우드 기사 이탈확률이 미매칭 비율의 함수다.
  ar5iv 본문 4.2절에서 확인한 식: `p = p_high · (미매칭 CD 비율) + p_low · (1 − 미매칭 CD 비율)`, 기본값 `p_high = 1`, `p_low = 0.01`, 이탈 인원은 이항분포 [10].
- 차이 1: 결정변수가 **고정기사(FD) 채용 수**(스칼라 수량)다. 잉여배분 몫이 아니다.
- 차이 2: VFA가 **FD 차원의 분리형 구간선형(PL-VFA, 오목성 유지)** 이다. 본문 3.3.2절 인용: "We seek for a piecewise linear approximation of V along the FD dimension" [10].
  즉 우리 프로젝트가 "기존 ADP의 표준 근사구조"로 지목한 바로 그 형태이며, 비분리형 학습 VFA나 그래프 정보는 없다.
- 차이 3: 반응이 **미매칭 비율의 선형 혼합**이고 금전(잉여율)의 로지스틱이 아니다.
- 차이 4: 참여자가 동질적 **집계 인원수**다. 개별 프로필·거래 연결 구조·다품목 묶음이 없다.
- 차이 5: 계산 실험 비교군이 PL-VFA vs **근시안(MY, 당기 수요를 채울 만큼 FD를 항상 고용)** 두 가지다 [10].

### (2) Chen, Zheng, Ke, Yang (2020), *Transportation Research Part B* — A2에 가장 가까움 [12]
- 공통점: **수수료율(commission rate)을 기간별 결정변수로** 두고 서지가격·인센티브와 함께 **ADP 기반 non-myopic 알고리즘**으로 푼다.
  전략이 승객·공차의 **도착률**에 영향을 주므로 참여가 결정의존이다 [12].
- 차이 1: 수수료율이 **시장 전체 단일 스칼라**다. 주문자·공급자 **개별**에게 잉여를 나누는 결정이 아니다.
- 차이 2: 반응이 집계 **도착률**이다. 개별 재참여확률이 아니므로 "참여자 i를 잃는 손해 c_i"라는 개념이 성립하지 않는다.
- 차이 3: 다품목 all-or-nothing 묶음, 품목별 용량 배정, MILP 매칭이 없다.
- 주의: 이 논문의 내용은 검색 요약을 경유해 확보했다(ScienceDirect 403, OpenAlex 초록 없음). **원문 확인 전까지 medium confidence.**

### (3) Musalem, Olivares, Yung (2023), *Operations Research* [9]
- 공통점: 운영 결정이 **에이전트 유지(이직)** 를 통해 장기 성과에 영향을 준다는 트레이드오프를 정면으로 다루고, 이직 모형을 **실데이터로 추정**한다(A3 실증 앵커).
- 차이: 결정변수가 **용량(스태핑) 수준**이다. 매칭도 잉여배분도 없고, 학습 VFA를 최적화에 넣지 않는다.

### (4) Balseiro, Lin, Mirrokni, Paes Leme, Zuo (2017), NeurIPS [13]
- 공통점: **수익배분 몫 α가 기간별 결정변수**다. 반복 상황을 이용해 제약을 기간 간에 배분한다.
- 차이(결정적): 판매자 참여가 **기대값 IR 제약**(판매자가 기회비용 c 이상 받아야 함)으로 강제된다.
  참여자가 확률적으로 떠나거나 참여자 집합이 변하지 않으므로 "배분 → 재참여확률 → 미래 참여자 구성" 되먹임이 없다.

보조: Chen & Xu (2026, 프리프린트) [17]은 "배분이 미래 참여를 바꾸므로 공정 배분이 장기 이익에 대한 투자가 된다"는 논리 구조가 우리와 가장 닮았다.
다만 배분 대상이 금전 잉여가 아니라 **디스패치 수량**이고, 결정론적 DP이며 학습 VFA가 없고 미심사 프리프린트다.

---

## 3. A2에 대한 직답

**부분적으로 존재한다. 그러나 "잉여배분 몫 + 확률적 재참여 + 참여자 집합 전이"를 동시에 갖춘 동적 모형은 찾지 못했다.**

찾은 것을 정확히 구분하면 세 갈래다.

1. **잉여배분 몫을 기간별 결정변수로 둔 동적 모형은 있다** — Balseiro et al. (NeurIPS 2017) [13]. 단, 참여는 기대값 IR 제약이며 참여자 집합은 불변이다.
2. **수수료율을 기간별 결정변수로 두고 ADP로 푼 모형도 있다** — Chen et al. (TR-B 2020) [12]. 단, 단일 스칼라 수수료율이고 반응이 집계 도착률이다.
3. **참여자 이탈이 결정에 내생인 ADP도 있다** — Luy et al. (POM 2024) [10]. 단, 결정변수가 채용 수이고 이탈이 미매칭 비율에 반응한다(금전 배분에 반응하지 않는다).

**빈 칸(갭)**: 참여자 **개별**에게 가는 잉여 몫을 기간별 결정변수로 두고, 그 몫이 개별 **재참여확률**을 통해
**다음 기간 참여자 집합과 거래 연결 구조**를 바꾸며, 이를 하나의 매칭 MILP 안에서 함께 결정한 모형.
(1)·(2)는 배분을 결정하지만 집합을 바꾸지 않고, (3)은 집합을 바꾸지만 배분을 결정하지 않는다.

확인에 쓴 쿼리(0건 또는 무관 결과만 반환):
- OpenAlex `title_and_abstract.search`
  - `logistics platform surplus allocation participant retention dynamic` → **count=0**
  - `all-or-nothing bundle order platform dynamic allocation supplier` → **count=0**
  - `B2B platform dynamic matching suppliers buyers` → **count=0**
  - `platform revenue sharing participant retention dynamic program` → count=1 (무관: e-learning 프라이버시)
  - `endogenous participant set dynamic optimization platform` → count=4 (전부 무관: 정밀의료·V2G·마케팅·바이오마커)
- WebSearch `"endogenous" participant pool dynamic program platform "surplus allocation" OR "profit allocation" per-period decision matching`
  → 관련 히트는 Chen & Xu(VPP 프리프린트)와 Okada(장기 배분 프리프린트)뿐.
- WebSearch `platform commission rate as dynamic control variable seller retention Markov decision process marketplace`
  → 학술 결과 없음. 상위 결과가 전부 마켓플레이스 SEO 블로그였다. (이 자체를 공백 신호로 읽는 것은 **추론**이다.)
- WebSearch `dynamic revenue sharing rate decision platform multi-period supplier participation Management Science M&SOM`
  → 수익배분은 **정태 계약·설계** 문헌([14][15])으로만 나온다.

---

## 4. A1 / A3 / A4 / A5

### A1. 당기 결정이 다음 기간 참여자 집합을 바꾸는 동적 최적화
직접 해당하는 연구는 **소수**이며 세 계보 중 하나에 속한다.
- **인력계획 계보**: Gans & Zhou 2002 [22] → Arlotto·Chick·Gans 2014 [21] → Luy et al. 2024 [10].
  채용·유지 결정이 인력 구성을 바꾸는 MDP. Luy et al.에서 처음으로 이탈확률이 **운영 결정(매칭)** 에 내생이 된다 [10].
- **플랫폼 경제학 계보**: Taylor 2018 [1], Cachon et al. 2017 [2], Guda & Subramanian 2019 [3], Benjaafar et al. 2022 [5], Bhargava et al. 2022 [14].
  참여가 가격·임금·배분율에 내생이지만 **정태 균형**이며 기간 간 상태 전이가 없다.
- **동적 매칭 계보**: Hu & Zhou 2022 [6], Aouad & Sarıtaç 2022 [7], Akbarpour et al. 2020 [20], You & Vossen 2024 [11].
  참여자가 오고 떠나지만 **이탈률이 외생**이며, 플랫폼은 매칭 시점으로만 구성에 영향을 준다. 금전 배분 결정이 없다.
- 수요 측 내생성을 다룬 프리프린트: Chen·Ulmer·Thomas 2025 [18](서비스품질 → 지역 수요 성장), Chen & Xu 2026 [17](배분 → 참여상태).
- 방법론 라벨: 이런 전이는 **decision-dependent(endogenous) uncertainty** 로 불린다 [23].

### A2. 결정변수 — §3 참조.

### A3. 이탈·재참여 반응 함수의 형태와 추정·강건성
관찰된 함수형은 다양하고, **로지스틱을 쓴 사례는 이번 조사에서 확인하지 못했다.**
- **유보값·임계형**: Taylor 2018 [1], Cachon et al. 2017 [2](기회비용·유보임금 임계), Schur & Winheller 2024 [16](수락 임계값을 확률변수로), Liu et al. 2023 [19](임계 기반 인센티브, 서지만 확인).
- **선형 혼합 확률**: Luy et al. 2024 [10]. 미매칭 비율에 대한 p_high/p_low **두 점 보간**이다.
  우리가 로지스틱을 두 점(잉여 0 → 0.5, 기준 잉여율 → 0.85)으로 고정한 방식과 "두 점으로 반응을 고정한다"는 점에서 유사하다.
- **오목·포화형**: Chen & Xu 2026 [17], `g(x) = 1 − exp(−ηx)`.
- **집계 도착률 반응**: Chen et al. 2020 [12].
- **데이터 추정 이직·생존 모형**: Musalem et al. 2023 [9], Xu et al. 2025 [25](반복 생존 + Transformer로 **학습된 반응**).
- **강건성**: 반응함수 오지정(misspecification)에 대한 강건성을 플랫폼 유지 맥락에서 정면으로 다룬 연구를 **찾지 못했다.**
  검색어: `robustness to misspecified behavioral response function dynamic platform optimization sensitivity retention model alternative functional forms operations`,
  `robust Markov decision process uncertain customer response function platform incentive retention wrong model evaluation`
  → 일반 robust MDP(전이확률 불확실성 집합) 문헌만 반환되고 참여·유지 반응에 특화된 연구는 없었다.
  우리 실험 2(바뀐 반응에 재학습 / 로지스틱으로 학습해 다른 반응에서 평가)는 이 공백에 놓인다.
- 반박 근거: Masorgo et al. 2026 [26]은 크라우드십핑 기사 반응이 **금전만의 함수가 아니라** 운영 특성에 좌우된다고 본다(서지만 확인).
  "재참여확률 = 로지스틱(잉여율)" 단일 변수 가정에 대한 실증적 반론으로 인용될 수 있다.

### A4. 미들마일 / B2B 원자재 / 5PL / 다품목 묶음 조달 맥락
**없다.** 이 맥락에서 참여자 구성 내생성을 동적으로 다룬 연구를 찾지 못했다.
- 5PL 문헌(OpenAlex `fifth-party logistics` 72건)은 **개념·역량·디지털 전환 논의**이며 최적화 모형이 아니다 [32][33].
  WebSearch `"fifth party logistics" 5PL platform matching optimization model` 역시 학술 결과 대신 업계 설명 페이지만 반환했다.
- OpenAlex `title_and_abstract.search`
  - `middle-mile transportation platform matching` → **count=2** (무관: EPFL 학위논문 1건, Kotler 저서 1건)
  - `middle mile consolidation hub dynamic optimization platform` → **count=0**
  - `all-or-nothing bundle order platform dynamic allocation supplier` → **count=0**
  - `B2B platform dynamic matching suppliers buyers` → **count=0**
- 인접 문헌은 **협력 운송의 이익배분**인데 Shapley 값 등 **협조게임·정태** 접근이고 동적 MDP나 참여자 집합 전이가 없다
  (WebSearch 수준에서만 확인했고 개별 논문은 서지조차 검증하지 않았으므로 증거 표에 넣지 않았다).
- 결론: A4는 **명확한 공백**이다. 단, 근거는 위 쿼리들이며 색인 한계(INFORMS 초록 미수록이 많다)를 감안해야 한다.

### A5. 성과 지표·벤치마크와 우리 비교군의 약점
문헌 표준(확인된 것만):
- **근시안(myopic)**: Luy et al. [10](MY 채용정책), Chen et al. [12](non-myopic ADP vs 근시안), Chen & Xu [17](greedy 이익최대).
- **규칙·휴리스틱**: Aouad & Sarıtaç [7](배치 알고리즘), Hu & Zhou [6](휴리스틱 정책), Chen & Xu [17](엄격 공정성 규칙).
- **LP/ALP 상한과 최적성 격차**: You & Vossen [11]은 ALP 재정식화로 **실행 가능 정책과 상한을 동시에** 얻어 최적성 격차를 보고한다.
  Aouad & Sarıtaç [7]은 **LP 벤치마크 + 상수비 근사 보증**을 제시한다.
- **완전정보(후견) 상한**: Luy et al. 초록이 "완전한 미래 정보를 가진 lookahead 정책"을 비교군으로 언급한다 [10]. ADP 일반에서 표준 어휘다 [24].
- **유체 극한**: 플랫폼·ride-hailing 계보에서 널리 쓰이지만, 이번 조사에서 특정 논문의 본문으로 **직접 확인하지 않았다.** 단정하지 않는다.

우리 비교군의 약점:
1. **상한이 없다.** 우리 비교군은 규칙 기반·근시안·선형 VFA(모두 정책)뿐이다. 문헌 표준은 **LP/ALP 완화 상한** 또는
   **후견(완전정보) 상한** 대비 격차를 보고한다 [11][7][24]. `match/hindsight.py`가 있어도 결과 표에 상한 대비 격차를 싣지 않으면 문헌 기준으로 약하다.
2. **근사 보증이 없다.** [7][11]은 이론 보증이나 검증된 격차를 제시한다. 소규모 예비 실험임을 한계로 명시해야 한다.
3. 규칙 기반을 4×4 격자로 튜닝하는 것은 적절하다(비교군 약화 방지). 다만 [10]처럼 **완전정보 lookahead**를 추가하면 문헌 수준에 가까워진다.
4. 우리가 **더 보수적인 점**: [10][12]는 근시안 대비 개선이 주된 근거인데, 우리는 근시안을 "참고용 하한"으로 내리고
   튜닝된 규칙 기반을 주 비교 기준으로 삼는다.

---

## 5. 인용 그래프에서 관찰한 것 (갭 신호)

수집 방법: 씨앗 8편(플랫폼·매칭 계보: Taylor, Cachon, Guda·Subramanian, Besbes 외, Benjaafar 외, Hu·Zhou, Afèche 외, Aouad·Sarıtaç)과
씨앗 4편(동적 최적화 계보: Luy 외, Schur·Winheller, You·Vossen, Chen 외 TR-B)의 참조 목록을 각각 교차해 공통 참조를 계수했다.

1. **플랫폼·매칭 씨앗 8편의 공통 참조는 6편뿐이다.** ≥3편이 공유한 참조는
   Cachon et al. 2017(5회), Taylor 2018(5회), Bai et al. 2018(4회), Bimpikis et al. 2017(3회), Zervas et al. 2017(3회), Fraiberger & Sundararajan 2015(3회).
   즉 이 군집의 공통 핵은 **정태 가격·임금 균형 모형**이다. 동적 참여자 전이를 다룬 공통 참조는 하나도 없다.
2. **동적 최적화 씨앗 4편의 공통 참조는 단 2편이다**: Powell의 *Approximate Dynamic Programming*(3회)과 Cachon et al. 2017(3회).
   ADP 계보와 플랫폼 경제학 계보를 잇는 고리는 **"Cachon et al.을 동기부여로 인용한다"** 수준에 그친다.
   → **갭 신호 1**: 참여를 내생화하는 군집(정태 균형)과 동적으로 VFA를 학습·근사하는 군집(ADP 물류)이 서로의 방법을 쓰지 않는다.
3. **동적 매칭 이론은 학습 VFA와 거의 만나지 않는다.** Aouad & Sarıtaç(OR 2022)를 인용한 문헌 중
   제목·초록에 `reinforcement learning`이 있는 것은 **1편**(You & Vossen 2024 [11]), `value function`이 있는 것은 **2편**뿐이다.
   → **갭 신호 2**.
4. **유지(retention) 실증 군집은 최적화로 이어지지 않는다.** Musalem et al.(OR 2023)을 인용한 20편은 거의 전부
   **행동·인력 실증**(근무 스케줄 변동성과 이직, 업무부하와 고숙련 인력 이탈, 콜센터 긱 이코노미)이다.
   이 중 동적 매칭·잉여배분 최적화로 넘어간 후속 연구는 없다.
   → **갭 신호 3**: 반응함수를 추정하는 군집과 그 반응을 최적화에 넣는 군집이 분리되어 있다.
5. **Luy et al.의 참조 목록이 갭의 위치를 보여준다.** 그 참조는 (a) 크라우드소싱 배송 라우팅(Arslan 외, Archetti 외, Dayarian·Savelsbergh, Ulmer·Savelsbergh)과
   (b) **인력계획·이직 계보**(Gans·Zhou 2002, Arlotto 외 2014, Ahn 외 2005, Jaillet·Loke·Sim 2022)의 결합이다.
   **잉여배분·수익공유 문헌은 한 편도 인용되지 않는다.**
   → **갭 신호 4**: "이탈 내생 + ADP"를 하는 사람들은 수익배분 문헌을 읽지 않고, 수익배분을 하는 사람들([13][14][15])은 ADP를 쓰지 않는다.
   우리 연구가 놓이는 자리가 정확히 이 교차점이다.
6. **수익배분 문헌 내부의 분단**: 정태 계약(Cachon & Lariviere 2005 [15]) → 플랫폼 차등배분 설계(Bhargava 외 2022 [14])는 같은 정태 전통이고,
   동적 수익배분(Balseiro 외 2017 [13])은 광고경매·온라인 알고리즘 전통이다. 셋 다 참여자 집합 전이를 다루지 않는다.

---

## 6. 갭 교차표

행 = 선행연구, 열 = 특성. O = 있음, X = 없음.

| 연구 | 다기간 동적 | 참여자 집합 내생 | 잉여배분이 결정변수 | 개별(참여자별) 배분 | 다품목 묶음 | 학습된 VFA | 매칭을 IP로 |
|---|---|---|---|---|---|---|---|
| Luy 외 2024 [10] | O | O | X | X | X | X (분리형 PL-VFA) | X (채용 수) |
| Chen 외 2020 TR-B [12] | O | O (도착률) | O (수수료율) | X | X | X (ADP, 형태 미확인) | X |
| Balseiro 외 2017 [13] | O | X (IR 제약) | O | O (판매자별 α) | X | X | X |
| Musalem 외 2023 [9] | 부분 | O | X | X | X | X | X |
| Aouad·Sarıtaç 2022 [7] | O | 부분(외생 이탈) | X | X | X | X (분석적 VFA) | X |
| You·Vossen 2024 [11] | O | X | X | X | X | X (ALP) | O |
| Hu·Zhou 2022 [6] | O | X | X | X | X | X | X |
| Bhargava 외 2022 [14] | X | O | O | O (규모별 차등) | X | X | X |
| Chen·Xu 2026 [17] | O | O | X (수량 배분) | O | X | X | X |
| Chen·Ulmer·Thomas 2025 [18] | O | O (수요 측) | X | X | X | O (RL) | X |
| **우리 설정** | **O** | **O** | **O** | **O** | **O** | **O** | **O (MILP)** |

빈 칸이 겹치는 곳: **"잉여배분이 결정변수 + 개별 배분 + 참여자 집합 내생 + 다품목 묶음 + 학습 VFA를 IP에 결합"** 을 동시에 만족하는 행이 없다.

---

## 7. 미해결 · 확인 불가 (도서관 접속으로 확인할 질문)

1. **Chen, Zheng, Ke, Yang (2020) TR-B** [12] — ScienceDirect 403. 확인 질문:
   (a) 수수료율이 기간마다 재결정되는 상태의존 정책인가, 아니면 시간대별 사전 프로파일인가?
   (b) ADP의 가치함수 근사 형태(선형 기저? 룩업 테이블?)
   (c) 도착률 반응함수의 구체적 함수형과 추정 근거
   (d) 비교군에 규칙 기반이나 후견 상한이 포함되는가?
2. **Musalem, Olivares, Yung (2023) OR** [9] — INFORMS 403. 확인 질문:
   (a) 이직 반응의 추정 함수형(비례위험? 로짓?)과 설명변수(가동률인가 소득인가)
   (b) 최적화가 동적계획인가 시뮬레이션 기반 정태 최적화인가
   (c) 후견 상한 비교가 있는가
3. **Luy, Hiermann, Schiffer (2024) POM** [10] — arXiv 판으로 모델·실험 절은 읽었다. 남은 질문:
   (a) 완전정보 lookahead 벤치마크의 정확한 정의와 결과 수치(초록에 "up to [Formula]"로 가려져 있다)
   (b) PL-VFA의 분리 가능성 가정이 어디서 필요한지 (우리 B3·B4 논의와 직결)
4. **Liu, Xu, Vignon, Yin, Qin, Li (2023) TR-C** [19] — 초록 미확보. 확인 질문: 임계 기반 인센티브의 반응함수 형태, 기간 간 재참여를 다루는지.
5. **Delarue, Lian, Qin (2026) SSRN** [31] — SSRN 403. 확인 질문: 보수(pay) 결정이 기사 유지를 통해 미래에 영향을 주는 동적 모형인가?
   그렇다면 A2의 답이 바뀔 수 있다. **우선 확인 1순위.**
6. **Lei, Jasin, Wang, Deng (2020) SSRN** [29] — 초록 미확보. 확인 질문: 인력 확보 결정이 참여자 집합 전이를 명시적으로 모형화하는가, 반응함수는 무엇인가.
7. **Masorgo 외 (2026) JOM** [26] — 초록 미확보. 확인 질문: 기사 유지에 금전 외 어떤 운영 특성이 유의한가
   (우리 "로지스틱(잉여율)" 단일 변수 가정에 대한 반박 강도 평가용).
8. **Afèche, Liu, Maglaras (2023) M&SOM** [8] — 초록 전문 미확보. 확인 질문: 유체 극한을 쓰는지, 기사 진입·퇴출이 기간 간 상태로 모형화되는지.
9. **협력 운송 이익배분(Shapley) 문헌** — WebSearch 수준에서만 확인해 증거 표에 넣지 않았다.
   확인 질문: 이 문헌 중 **다기간에 걸쳐 배분 결정이 다음 기간 연합 구성(참여자 집합)을 바꾸는** 모형이 있는가?
   (있다면 A2·A4의 답이 약해진다.)
10. **유체 극한 벤치마크** — 특정 논문 본문에서 직접 확인하지 못했다. A5의 해당 항목은 현재 근거가 약하다.

### 반박 가능성 (누가 "이미 있다"고 말할 수 있는가)
- "잉여배분을 동적 결정변수로 둔 건 이미 있다" → **Balseiro 외 2017** [13] 또는 **Chen 외 2020** [12].
  우리 방어: [13]은 참여자 집합이 불변이고 참여가 하드 IR 제약, [12]는 단일 스칼라 수수료율 + 집계 도착률 반응.
- "이탈 내생 ADP는 이미 있다" → **Luy 외 2024** [10].
  우리 방어: 결정변수가 채용 수, VFA가 분리형 구간선형, 반응이 금전이 아니라 미매칭 비율.
- "참여가 배분에 내생인 동적 모형은 이미 있다" → **Chen & Xu 2026** [17].
  우리 방어: 미심사 프리프린트, 배분 대상이 수량, 결정론적 DP, 학습 VFA·묶음·그래프 없음.
- "5PL·미들마일 연구도 있다" → [32][33]. 우리 방어: 개념·역량 논의이며 최적화 모형이 아니다.

---

## 8. Coverage Status

직접 확인한 것:
- OpenAlex로 씨앗 12편의 참조·피인용을 순회하고 공통 참조를 계수했다(§5의 숫자는 그 계수 결과다).
- 전문(해당 절) 직접 읽음: Luy 외 2024 arXiv 판(이탈확률 함수형·PL-VFA·벤치마크), Chen & Xu 2026 arXiv(상태 전이식·g 함수·벤치마크),
  Balseiro 외 2017 NeurIPS PDF(결정변수·IR 제약·벤치마크).
- 초록 전문 직접 확보(OpenAlex 또는 출판사 페이지): [1][2][4][5][6][7][9][11][14][15][16][20][27][28][34] 등.
- A4 부정 증거를 OpenAlex 0건 쿼리로 남겼다(§4 A4).

불확실한 것:
- 저널 사분위를 **하나도 직접 확인하지 않았다**. 모두 "사분위 미확인"으로 표기했다.
- Chen 외 2020 TR-B [12]의 모델 세부는 검색 요약 경유다. A2의 핵심 근거이므로 **원문 확인 전에는 medium confidence**로 다뤄야 한다.
- 유체 극한 벤치마크(A5)의 근거가 약하다.
- 프리프린트 [17][18][25][27][34]는 미심사임을 명시했다.
- [34]는 arXiv 페이지에서 저자를 확보하지 못했다. 저자 미확인 상태로 두었다.

완료하지 못한 것:
- SSRN 403으로 [29][31]의 초록을 얻지 못했다. [31](Delarue·Lian·Qin, pay strategies)은 A2의 답을 바꿀 수 있는 유일한 미확인 후보다.
- 협력 운송 이익배분(Shapley) 문헌을 개별 논문 수준으로 검증하지 않았다(§7-9).
- INFORMS·ScienceDirect 페이월(403)로 [9][12][19][26]의 본문을 읽지 못했다.

---

## 9. Sources

1. Taylor, T. A. (2018) *On-Demand Service Platforms*, M&SOM — https://doi.org/10.1287/msom.2017.0678
2. Cachon, G. P., Daniels, K. M., Lobel, R. (2017) *The Role of Surge Pricing on a Service Platform with Self-Scheduling Capacity*, M&SOM — https://doi.org/10.1287/msom.2017.0618
3. Guda, H., Subramanian, U. (2019) *Your Uber Is Arriving: Managing On-Demand Workers Through Surge Pricing, Forecast Communication, and Worker Incentives*, Management Science — https://doi.org/10.1287/mnsc.2018.3050
4. Besbes, O., Castro, F., Lobel, I. (2021) *Surge Pricing and Its Spatial Supply Response*, Management Science — https://doi.org/10.1287/mnsc.2020.3622
5. Benjaafar, S., Ding, J.-Y., Kong, G., Taylor, T. (2022) *Labor Welfare in On-Demand Service Platforms*, M&SOM — https://doi.org/10.1287/msom.2020.0964
6. Hu, M., Zhou, Y. (2022) *Dynamic Type Matching*, M&SOM — https://doi.org/10.1287/msom.2020.0952 (arXiv https://arxiv.org/abs/1811.07048)
7. Aouad, A., Sarıtaç, Ö. (2022) *Dynamic Stochastic Matching Under Limited Time*, Operations Research — https://doi.org/10.1287/opre.2022.2293
8. Afèche, P., Liu, Z., Maglaras, C. (2023) *Ride-Hailing Networks with Strategic Drivers*, M&SOM — https://doi.org/10.1287/msom.2023.1221
9. Musalem, A., Olivares, M., Yung, D. (2023) *Balancing Agent Retention and Waiting Time in Service Platforms*, Operations Research 71(3):979–1003 — https://doi.org/10.1287/opre.2022.2418
10. Luy, J., Hiermann, G., Schiffer, M. (2024) *Strategic Workforce Planning in Crowdsourced Delivery With Hybrid Driver Fleets*, Production and Operations Management — https://doi.org/10.1177/10591478241268602 (arXiv https://arxiv.org/abs/2311.17935)
11. You, F., Vossen, T. W. M. (2024) *An Approximate Dynamic Programming Approach to Dynamic Stochastic Matching*, INFORMS Journal on Computing — https://doi.org/10.1287/ijoc.2021.0203
12. Chen, X., Zheng, H., Ke, J., Yang, H. (2020) *Dynamic optimization strategies for on-demand ride services platform: Surge pricing, commission rate, and incentives*, Transportation Research Part B 138:23–45 — https://doi.org/10.1016/j.trb.2020.05.005
13. Balseiro, S., Lin, M. S., Mirrokni, V., Paes Leme, R., Zuo, S. (2017) *Dynamic Revenue Sharing*, NeurIPS 2017 — https://papers.nips.cc/paper/2017/file/cb8acb1dc9821bf74e6ca9068032d623-Paper.pdf (SSRN https://doi.org/10.2139/ssrn.2956715)
14. Bhargava, H. K., Wang, K., Zhang, X. (2022) *Fending Off Critics of Platform Power with Differential Revenue Sharing: Doing Well by Doing Good?*, Management Science — https://doi.org/10.1287/mnsc.2022.4545
15. Cachon, G. P., Lariviere, M. A. (2005) *Supply Chain Coordination with Revenue-Sharing Contracts: Strengths and Limitations*, Management Science — https://doi.org/10.1287/mnsc.1040.0215
16. Schur, R., Winheller, K. (2024) *Optimizing last-mile delivery: a dynamic compensation strategy for occasional drivers*, OR Spectrum — https://doi.org/10.1007/s00291-024-00796-6
17. Chen, L., Xu, B. (2026, 프리프린트) *Fairness as an Investment: Dynamic Participation and Long-Run Profit in Virtual Power Plants*, arXiv — https://arxiv.org/abs/2606.02820
18. Chen, X., Ulmer, M. W., Thomas, B. W. (2025, 프리프린트) *To Start Up a Start-Up — Embedding Strategic Demand Development in Operational On-Demand Fulfillment via Reinforcement Learning with Information Shaping*, arXiv — https://arxiv.org/abs/2504.05633
19. Liu, T., Xu, Z., Vignon, D., Yin, Y., Qin, Z. T., Li, Q. (2023) *Threshold-based incentives for ride-sourcing drivers: Implications on supply management and welfare effects*, Transportation Research Part C — https://doi.org/10.1016/j.trc.2023.104323
20. Akbarpour, M., Li, S., Oveis Gharan, S. (2020) *Thickness and Information in Dynamic Matching Markets*, Journal of Political Economy — https://doi.org/10.1086/704761
21. Arlotto, A., Chick, S. E., Gans, N. (2014) *Optimal Hiring and Retention Policies for Heterogeneous Workers Who Learn*, Management Science — https://doi.org/10.1287/mnsc.2013.1754
22. Gans, N., Zhou, Y.-P. (2002) *Managing Learning and Turnover in Employee Staffing*, Operations Research — https://doi.org/10.1287/opre.50.6.991.343
23. Hellemo, L., Barton, P. I., Tomasgård, A. (2018) *Decision-dependent probabilities in stochastic programs with recourse*, Computational Management Science — https://doi.org/10.1007/s10287-018-0330-0
24. Powell, W. B. (2011) *Approximate Dynamic Programming: Solving the Curses of Dimensionality*, 2nd ed., Wiley — https://doi.org/10.1002/9781118029176
25. Xu, S., Zhang, Y., Miller, E. J. (2025, 프리프린트) *Frailty-Aware Transformer for Recurrent Survival Modeling of Driver Retention in Ride-Hailing Platforms*, arXiv — https://arxiv.org/abs/2511.19893
26. Masorgo, N., Dobrzykowski, D. D., Tang, C. S., Fugate, B. S. (2026) *There Is More to Crowdshipping Than Money*, Journal of Operations Management — https://doi.org/10.1002/joom.70046
27. Okada, G. (2026, 프리프린트) *Dynamic Pooling and Regional Participation in Deceased-Donor Organ Allocation*, arXiv — https://arxiv.org/abs/2609.18147
28. Dogan, M., Jacquillat, A. (2025) *On-Demand Service Sharing via Collective Dynamic Pricing*, M&SOM — https://doi.org/10.1287/msom.2024.1301
29. Lei, Y., Jasin, S., Wang, J., Deng, H. (2020, 워킹페이퍼) *Dynamic Workforce Acquisition for Crowdsourced Last-Mile Delivery Platforms*, SSRN — https://doi.org/10.2139/ssrn.3532844
30. Bimpikis, K., Candogan, O., Sabán, D. *Spatial Pricing in Ride-Sharing Networks*, Operations Research — https://openalex.org/W2766210608 (주의: OpenAlex가 이 레코드에 붙인 DOI는 ACM EC 2017 판(10.1145/3106723.3106728)이다. Operations Research 게재판 DOI는 **미확인**)
31. Delarue, A., Lian, Z., Qin, Z. T. (2026, 워킹페이퍼) *Learning Pay Strategies with Small Samples in Gig Economy Platforms*, SSRN — https://doi.org/10.2139/ssrn.6197638
32. Hosie, P. J., Sundarakani, B., Tan, A. W. K. (2012) *Determinants of fifth party logistics (5PL): service providers for supply chain management*, IJLSM — https://doi.org/10.1504/ijlsm.2012.049700
33. Nicoletti, B., Appolloni, A. (2024) *Digital transformation in ecosystems: integrated operations model and its application to fifth-party logistics*, JGOSS — https://doi.org/10.1108/jgoss-04-2023-0024
34. (저자 미확인, 2026, 프리프린트) *Not All Matches Are Equally Valuable: An Online Experiment of Retention-Focused Recommendation in a Job-Matching Platform*, arXiv — https://arxiv.org/abs/2609.01652
