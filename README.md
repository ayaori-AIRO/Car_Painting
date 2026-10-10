# Car Painting — 자동차 부위 세그멘테이션

YOLO11n-seg로 자동차 24개 부위를 구분하고, 앞휀더 라벨 정제 및 데이터 증강의 효과를 평가하는 프로젝트입니다.

## 구성

- `scripts/`: 학습, 증강 미리보기, 증강 실험, 사진 예측 코드
- `configs/dataset.example.yaml`: 클래스 목록과 데이터셋 구조 예시
- `reports/`: 전체 클래스 및 휀더 평가 결과와 이미지 개수
- `dataset/`, `runs/`, `.venv/`: 로컬 전용이며 Git에 포함하지 않습니다.

## 환경 준비

Windows PowerShell, Python 3.13, RTX 4060 Laptop 8GB 환경에서 실행했습니다.
가상환경 생성 후 GPU와 드라이버에 맞는 CUDA PyTorch를 먼저 설치하고 다음 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

AlbumentationsX는 AGPL-3.0 또는 별도 상용 라이선스로 제공됩니다. 사용 조건은 해당 프로젝트의 라이선스를 확인하세요.

## 데이터 준비

```text
dataset/carparts-seg/
  data.yaml
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
dataset/test/                 # 실제 차량 사진
```

라벨은 YOLO 세그멘테이션 폴리곤입니다. 휀더 클래스는 23번이며 앞휀더로 범위를 통일했습니다.
총 이미지 5,895장: 학습 4,694장, 검증 779장, 테스트 422장. 휀더 포함 이미지는 2,073장입니다.
데이터셋 파일은 저장소에 포함되지 않으며 별도로 준비해야 합니다.

## 실행

프로젝트 루트에서 실행합니다. 재학습은 GPU 자원을 사용하며 기존 모델이 필요한 실험은 해당 체크포인트를 준비해야 합니다.

```powershell
# 최초 모델 학습·평가
.\.venv\Scripts\python.exe scripts/train_and_test.py
# 조명·화질 미리보기
.\.venv\Scripts\python.exe scripts/preview_albumentationsx.py
# 조명·화질 실험: 각 8 epoch, 작업자 2개
.\.venv\Scripts\python.exe scripts/run_augmentation_experiments.py
# 노이즈·부분 가림 실험
.\.venv\Scripts\python.exe scripts/run_noise_occlusion_experiments.py
# dataset/test 사진에 최신 기본 모델 적용
.\.venv\Scripts\python.exe scripts/test_user_photos.py
```

추가 증강 실험의 초기 모델: `runs/carparts_fender_refresh_20261009/weights/best.pt`.
모든 클래스의 폴리곤을 유지합니다. 부분 가림은 가린 부분도 기존 정답으로 유지하는 실험입니다.
생성 데이터와 새 체크포인트는 `runs/`에 저장됩니다. 상태 화면 포트는 조명·화질 8791, 노이즈·가림 8792입니다.

## 완료한 실험 결과

동일한 원본 테스트 데이터에서 측정한 마스크 지표입니다.

| 모델 | 전체 mAP@50 | 전체 mAP@50–95 | 휀더 AP@50 | 휀더 AP@50–95 |
|---|---:|---:|---:|---:|
| 기준 모델 | 73.3% | 58.4% | 90.1% | 61.1% |
| 조명 증강 | 87.3% | 70.6% | 92.3% | 64.3% |
| 화질 증강 | 88.0% | 70.8% | 90.9% | 62.5% |

조명: 밝기·대비와 감마. 화질: GaussianBlur, JPEG 압축, Downscale 중 하나.
각 모델은 원본 4,694장과 증강 4,694장으로 학습했습니다. 추가 학습 효과도 포함돼 있으므로 상승분 전체를 증강만의 효과로 해석하지 않습니다.
노이즈·부분 가림 실험은 별도로 진행 중입니다.

## 기존 실행 결과 위치

Git 관리 시작 전 실험 코드와 결과는 Codex 작업 폴더의 `outputs/`에서 생성됐습니다.
현재 실행 중인 실험은 그 코드를 계속 사용합니다. 이 저장소의 `scripts/`는 프로젝트 루트 기준 경로로 정리한 사본입니다.
학습 중인 실행과 새 실행을 동시에 시작하지 마세요.
