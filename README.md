# 딥페이크 변조 영상 탐지 시스템 구현 시도

데이콘 딥페이크 변조 영상 탐지 AI 경진대회 입상팀 **WeAreApe**가 공개한 코드와 학습 완료 가중치를 바탕으로 탐지 파이프라인을 구성해 본 학습용 프로젝트입니다.

> **참고:** 공개된 학습 완료 가중치로 추론과 Streamlit 데모만 구성했고, 직접 학습은 완료하지 못했습니다. 대회 입상은 원 저자(WeAreApe) 팀의 성과이며, 원본 코드의 저작권도 원 저자에게 있습니다.
(대회 링크: https://dacon.io/competitions/official/235655/overview/)

## 📂 파일 구성 및 역할
* [inference_dacon_data.py](<inference_dacon_data.py>): 최종 추론을 위한 메인 실행 파일 (Leaderboard 이미지 -> 모델 임베딩 -> SVM 분류 -> submission.csv 생성)
* [crop_face_dacon.py](<crop_face_dacon.py>): MTCNN 그래프(`mtcnn.pb`)를 사용하여 입력 이미지에서 고정밀 안면(Face) 영역을 추출하고 정렬하는 전처리 스크립트
* [data_generator_dacon.py](<data_generator_dacon.py>): 학습 및 추론 시 대용량 이미지 데이터를 배치 단위로 실시간 증강 및 로드하기 위한 커스텀 DataGenerator
* [triplet_loss_dacon.py](<triplet_loss_dacon.py>): TensorFlow 기반의 Semi-hard Triplet Loss (삼중항 손실) 함수 정의 파일

## 📂 필수 리소스 구성
추론 및 학습 코드를 실행하기 위해서는 아래 리소스 파일이 완비되어 있어야 합니다.
1. **`resource/mtcnn.pb`**: 안면 추출(MTCNN) 그래프 가중치 파일
2. **`triplet_effB4_ep06_BS28.pkl`**: SVM 분류기 직렬화 가중치 파일 (현재 프로젝트 루트에 존재)
3. **`triplet_effB4_ep06_BS28.hdf5`**: EfficientNet-B4 특징 추출용 네트워크 모델 파일 (현재 프로젝트 루트에 존재)

---

## 🛠️ 환경세팅
### version & device
* OS: Ubuntu 16.04 (또는 최신 Windows/Linux 호환)
* Python 3.7.8 (또는 TensorFlow 2.x 호환 버전)
* GPU 가속: CUDA 10.1 / CuDNN 7.6.5 (TensorFlow GPU 작동 환경 권장)

### 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 🚀 사용법

### 1. inference (추론)
* `inference_dacon_data.py`를 실행하여 테스트 데이터셋에 대한 딥페이크 탐지 판정을 진행하고 제출 파일을 생성합니다.

```bash
python inference_dacon_data.py <테스트_이미지_디렉토리> <sample_submission.csv_경로> <network_model_hdf5_경로> <svm_model_pkl_경로>
```

**실행 예시:**
```bash
python inference_dacon_data.py ./test/leaderboard ./resource/sample_submission.csv ./resource/triplet_effB4_ep06_BS28.hdf5 ./triplet_effB4_ep06_BS28.pkl
```

* **출력 결과**: 지정한 제출용 CSV 경로와 동일한 위치에 `*_out.csv` 제출용 파일이 자동으로 생성됩니다.

### 2. train (학습)

#### [안면 이미지 전처리]
* `crop_face_dacon.py`를 실행하여 원본 영상에서 딥페이크 판별용 안면 정렬 데이터셋을 구축합니다.

```bash
python crop_face_dacon.py <원본_이미지_디렉토리> <생성될_안면데이터_디렉토리>
```

### 3. Streamlit 웹 데모 앱 실행
* 웹 브라우저 상에서 이미지를 드래그 앤 드롭하여 간편하게 딥페이크 판별 결과를 확인하는 데모 앱입니다.
* 실행 전 `streamlit` 패키지가 필요합니다.

```bash
pip install streamlit
```

```bash
streamlit run app.py
```

* **주요 기능**:
  - 실시간 이미지 업로드 및 업로드 파일 시각화
  - MTCNN 및 OpenCV Haar Cascade 방식의 자동 폴백 안면 영역 크롭
  - 진짜 인물(REAL) vs 변조 영상(FAKE) 분류 신뢰도 차트(bar chart) 및 메트릭 시각화



