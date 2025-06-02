# Ai-Project
데이터셋 컬럼 정리는 '김은서'님의 정리로 채택

# Autoencoder.ipynb (11,12)학습 결과
총 샘플 수: 약 294,588개
수치형 사용 변수: credit_card_limit: 신용카드 한도, zipcode: 우편번호, transaction_dollar_amount: 거래 금액,
Long, Lat: 위도와 경도, credit_card: 카드 번호 (수치 처리됨)
전처리 후 전체 데이터 중 10,000개 샘플을 랜덤 추출하여 DBSCAN 실험에 사용

- Autoencoder 기반 이상치 탐지
입력 데이터를 인코딩-디코딩하여 재구성 오차를 측정
재구성 오차가 기준 임계값 보다 큰 샘플을 이상치로 판단
오차 기준은 정량적으로 계산된 0.00018을 사용

- DBSCAN (클러스터 밀도 기반 이상치 탐지 알고리즘)
이웃 거리(eps)와 최소 샘플 수(min_samples)를 기준으로 밀도가 낮은 점들을 이상치로 판단
다양한 파라미터 설정으로 결과 비교:
eps = [0.5, 1.0, 1.5]
min_samples = [5, 10]

- Autoencoder 학습 결과
에폭 수: 50
초기 손실 (Epoch 1): 1.3758
최종 손실 (Epoch 50): 0.0046
학습 초기에는 빠르게 손실이 감소했으며, 이후 안정적으로 수렴하여 과적합 없이 학습이 완료됨
손실 그래프는 아래와 같은 경향을 보임
빠른 감소 (1~10 epoch) → 점진적 감소 (11~20 epoch) → 수렴 상태 유지 (20 epoch 이후)
이러한 학습 결과는 Autoencoder가 정상 거래 패턴을 잘 학습했다는 것을 시사함.

- 이상치 탐지 결과
Autoencoder 결과 (재구성 오차 기준)
임계값 (Threshold): 0.00018
탐지된 이상치 수: 500건
전체 샘플 중 0.17% 가량이 이상치로 탐지됨
이는 정상 거래의 재구성은 정확하게 수행되었으며, 재구성 실패가 발생한 소수의 샘플이 정상 패턴과 다름을 의미

| eps | min\_samples | 이상치 수 |
| --- | ------------ | ----- |
| 0.5 | 5            | 168   |
| 0.5 | 10           | 249   |
| 1.0 | 5            | 139   |
| 1.0 | 10           | 182   |
| 1.5 | 5            | 133   |
| 1.5 | 10           | 146   |

eps = 0.5, min_samples = 10 설정에서 가장 많은 이상치(249건)가 탐지됨 -> 데이터에 분산이 크거나 이질적인 값이 많다
DBSCAN은 지리적 위치나 거래 금액 분포 상의 밀도 기반 이상치를 탐지

Autoencoder는 재구성 능력을 활용하여 정상 패턴을 학습하고, 이상 거래를 소수 정밀하게 탐지함
DBSCAN은 거리 기반 클러스터링을 통해 지리적·금액적으로 독립적인 이상치를 탐지하는 데 유리함
두 기법은 서로 다른 관점에서 이상치를 탐지하므로 결합 사용 시 탐지 성능 향상이 기대됨
향후 작업에서는 두 모델의 탐지된 이상치의 중복 여부 분석과 함께, 실제 라벨과의 비교 또는 이상치 시각화 등을 통해 정밀도/재현율 기반 평가가 필요





# Autoencoder(24).ipynb학습 결과

총 샘플 수: 2,512건
사용된 수치형 컬럼:
['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance']

-DBSCAN 이상치 탐지 결과
eps	min_samples	이상치 수
0.5	 5	 508
0.5	 10	 918
1.0	 5	 130
1.0	 10	 162
1.5  5	 45
1.5	 10	 96

적정값 판단: eps=1.0, min_samples=5일 때 이상치가 130건으로 나타났으며, 지나치게 민감하지도 무디지도 않은 수준으로 판단됨.

-Autoencoder 결과
Epoch 50까지 평균 재구성 손실 감소: 0.1872 → 0.1648

-MSE 기반 이상치 기준 (상위 5%)
Threshold: 0.51721
이상치 탐지 수: 126건

-모델 평가 및 해석
Autoencoder
손실이 Epoch마다 꾸준히 감소 → 모델이 정상 거래를 잘 학습함
이상치 탐지 기준을 데이터 기반으로 설정 (상위 5%) → 통계적으로 신뢰 가능한 방식
이상치 수: 126건 → 전체 거래의 약 5.01%, 실제 금융 이상거래 발생 비율과 유사

DBSCAN
다양한 eps/min_samples 조합 실험을 통해 민감도 조정 가능
eps=1.0, min_samples=5에서 130건 탐지 → Autoencoder 결과와 유사한 규모

문제점
eps, min_samples 선택이 민감하며 튜닝이 필요
고차원 데이터일수록 성능이 떨어질 수 있음
결과가 Autoencoder보다 다소 변동성이 큼 

## 21,22,23은 안돌아감


#RandomForest 분석결과

총 샘플 수: 284,807건
사용 변수: V1 ~ V28 (PCA 변환 특징), Amount (거래 금액, 정규화됨)
타겟 변수: Class (0: 정상 거래, 1: Fraud 거래)
Fraud 거래 비율: 약 0.17%

RandomForest 기반 이상치(Fraud 거래) 탐지
입력 데이터를 이용하여 정상/이상 거래를 분류하는 RandomForestClassifier 학습

주요 파라미터:
n_estimators: 100
max_depth: 10
random_state: 42

학습/검증 데이터 분할: 80% 학습, 20% 테스트

학습 결과
지표	값
Accuracy	99.92%
Precision	87.50%
Recall	76.19%
F1-score	81.29%

Confusion Matrix
Predicted Normal (0)	Predicted Fraud (1)
Actual Normal (0)	56,560	7
Actual Fraud (1)	9	29

Feature Importance 분석
가장 중요한 변수들:
V17
V14
V12
V10
→ PCA로 변환된 주요 특징들이 Fraud 거래 탐지에 기여
→ Amount 변수는 상대적으로 낮은 중요도

Precision-Recall Curve 결과
Fraud 거래 탐지 성능(Recall): 약 76%
Precision: 비교적 높아 잘못된 Positive 예측은 적은 편
비즈니스 측면에서 Fraud 탐지에 실질적으로 활용 가능

해석 및 결론
RandomForest는 불균형 데이터에서도 비교적 안정적인 이상치 탐지 성능을 보임
주요 Feature의 중요도를 확인 가능하여 모델 해석성이 높음
Recall은 약 76% 수준으로 Fraud 거래 탐지 시 어느 정도 유효한 결과 확보
Precision이 높아 잘못된 이상치 판정(정상 거래를 Fraud로 판단)도 적은 편

문제점 및 한계
데이터 불균형 문제 여전 → Recall을 높이기 위한 추가 기법 필요 (예: SMOTE, 부스팅 기반 모델 등)
Fraud 탐지에서 Recall이 중요한 만큼 XGBoost, LightGBM 등과 비교 실험 필요
RandomForest는 고차원 데이터에서는 속도 저하 가능성 존재

