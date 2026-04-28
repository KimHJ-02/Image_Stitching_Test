# image_stitching 프로젝트

## 개요

이 프로젝트는 여러 장의 사진을 입력으로 받아 파노라마(panorama)를 생성하는 간단한 image stitching 애플리케이션입니다. 기본 파이프라인은 특징점 검출 → 기술자(디스크립터) 계산 → 매칭 → 호모그래피 추정 → 이미지 워핑 및 블렌딩으로 구성됩니다. 추가로, 겹치는 영역에서 부드러운 합성을 위해 거리 변환(distance transform)을 이용한 페더링 기반 블렌딩을 적용했습니다.

## 주요 기능

- 이미지 로드: 프로젝트 실행 디렉터리의 `imgs` 폴더에서 이미지 파일(.jpg, .png 등)들을 자동으로 불러옵니다.
- 특징점 검출기: ORB(기본) 또는 SIFT(옵션)를 사용하여 키포인트와 디스크립터를 생성합니다.
- 매칭: KNN 매칭 + ratio test(0.75)를 사용하여 좋은 매칭을 골라냅니다.
- 호모그래피: RANSAC을 사용해 안정적인 호모그래피 행렬을 계산합니다.
- 워핑: `cv.warpPerspective`를 이용해 이미지를 기준 이미지에 맞춰 투영합니다.
- 블렌딩: 단순 평균 대신 distance transform 기반 페더링을 사용해 겹침 부분의 경계를 부드럽게 처리합니다.

## 필수 환경

- Python 3.6+
- numpy
- opencv-python (혹은 OpenCV 빌드, SIFT 사용 시 contrib/특정 빌드 필요)

설치 예 (가상환경 권장)

```bash
pip install numpy opencv-python
```

## 사용법

1. 이미지 준비
   - 프로젝트 루트에 `imgs` 폴더를 만들고 파노라마로 합성할 이미지들을 넣습니다. (파일명 순서가 영향을 줄 수 있으므로 필요 시 파일명 정렬)

2. 실행
   - python image_stitching.py [--output OUTPUT] [--detector {orb,sift}]

   - `--output`: 결과 파일명 (기본: panorama.jpg)
   - `--detector`: 'orb' 또는 'sift' (기본: orb)

예시:
```bash
python image_stitching.py --output result.jpg --detector sift
```

## 제한사항 및 주의점

- 최소 2장 이상의 이미지가 필요합니다. 그렇지 않으면 처리하지 않습니다.
- SIFT는 OpenCV 빌드에 따라 사용 불가할 수 있으며, 이 경우 프로그램이 예외를 던집니다.
- 현재 구현은 입력 경로가 `imgs` 폴더로 고정되어 있습니다. 다른 폴더를 사용하려면 코드의 `load_images` 호출부를 수정해야 합니다.
- 블렌딩은 distance transform을 사용하므로 큰 이미지에서는 메모리/시간 비용이 증가할 수 있습니다.

## 구현 요약(대학생 수준)

1) 특징점 검출 및 디스크립터 계산
   - ORB 또는 SIFT로 각 이미지의 키포인트(kp)와 디스크립터(desc)를 계산.

2) 매칭
   - BFMatcher(Hamming) 또는 FLANN(SIFT)로 KNN 매칭을 수행한 뒤 Lowe ratio test로 좋은 매칭만 선별.

3) 호모그래피 계산
   - 선별된 매칭으로 대응점 세트를 만들고 `cv.findHomography(..., cv.RANSAC)`으로 변환 행렬 H 추정.

4) 이미지 워핑 및 캔버스 계산
   - 투영된 모서리 좌표를 이용해 출력 캔버스 크기 및 오프셋을 계산하고 `cv.warpPerspective`로 이미지를 변환.

5) 블렌딩
   - 겹치는 영역에서 각 이미지 내부 픽셀의 distance transform 값을 계산해 가중치로 사용.
   - 가중치 비율로 선형 보간하여 부드러운 경계 생성.

## 추가 개선 제안

- 멀티밴드 블렌딩(피라미드 블렌딩)으로 고품질 결과 확보
- 이미지 정렬 및 노출 차 보정(색 보정, 히스토그램 정규화)
- 입력 경로를 다시 옵션으로 되돌리기 혹은 GUI/파일 선택기 추가
- 큰 이미지 처리를 위한 블록 처리 또는 GPU 가속

## 문의 및 변경 요청

README 파일을 수정하거나 실행 흐름(예: `imgs` 폴더가 없을 경우 자동 생성, 다른 경로 지원)을 추가하길 원하면 알려주세요.