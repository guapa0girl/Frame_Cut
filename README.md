# frame100 — 돼지 Pose Estimation 라벨링용 프레임 추출 파이프라인

CVAT에 올릴 돼지(군집 사육) 영상 프레임을 추출하는 프로젝트입니다.
영상은 그대로 라벨링하기엔 너무 많고 비슷한 프레임이 반복되므로,
**1) 균등 간격 샘플링 → 2) 자세(pose) 다양성 기반 후보 추출 → 3) 목표 장수만 FPS로 재선택**
세 단계로 프레임을 추출합니다.

## 폴더 구조

```
frame100/
├── day_videos/              # 주간 원본 영상 (CH010, .avi)
├── night_videos/            # 야간 원본 영상
├── models/
│   ├── pretrained/          # yolov8n-pose.pt(COCO 사람 keypoint), yolo26n.pt — 참고용, 실제 추출엔 미사용
│   └── trained/             # 돼지 전용 YOLO-pose 모델 (train1~4_best.pt)
│                             #   -> train4_best.pt 사용 중 (18 keypoints, class: pig)
├── extract_frame.py            # 1단계: 고정 간격(1분당 1장) 샘플링
├── extract_diverse_frames.py   # 2단계: 자세 다양성 기반 후보 샘플링 (threshold, YOLO-pose 이용)
├── select_diverse_subset.py    # 3단계: 후보 중 목표 장수만 FPS로 재선택 (서로 가장 다른 조합)
├── cvat_frames_day/             # 주간 결과물 (segment_001, 002, ... = CVAT task 단위)
│   ├── segment_001, segment_002, ...   # 최종적으로 CVAT에 올리는 결과
│   ├── _old_1min_sampling/              # (백업) 2단계 적용 전 1분 간격으로 만들었던 segment
│   ├── _all_threshold_candidates/       # (백업) 3단계 이전, threshold로만 거른 전체 후보
│   └── pose_pool_state.npz              # 지금까지 확정한 프레임들의 자세 벡터 (다음 배치 이어가기용)
└── cvat_frames_night/           # 야간 결과물
```

결과 폴더 구조는 `output_dir/segment_XXX/frame_NNNN__<원본영상명>_<식별정보>.jpg` 형태이며,
`segment_XXX` 폴더 하나가 CVAT task 하나에 대응하도록 100장 단위로 끊어서 저장합니다.

## 1단계 — `extract_frame.py`: 고정 간격 샘플링

가장 단순한 베이스라인 추출 방식.

- 지정 폴더의 모든 영상을 순서대로 훑으며 **N초(기본 60초, 즉 1분)마다 1장씩** 프레임을 뽑음
- AVI 특성상 끝부분 프레임 디코딩이 실패하는 경우가 있어, 실패 시 최대 30프레임까지 앞으로 물러나며 재시도 (`read_frame_safe`)
- 모든 영상에서 뽑힌 프레임을 시간 순서대로 이어붙인 뒤 100장씩 끊어서 `segment_001, segment_002, ...` 폴더에 저장
- `--start_folder` 옵션으로 세그먼트 번호를 원하는 지점부터 시작하도록 밀 수 있음 (앞 번호 폴더는 건드리지 않고 이어서 추출할 때 사용)

**한계**: 1분 간격은 돼지가 계속 눕거나 자는 구간에서는 거의 똑같은 프레임을 계속 뽑고,
반대로 활동이 많은 구간의 다양한 자세는 놓칠 수 있음. → 2단계로 보완.

## 2단계 — `extract_diverse_frames.py`: 자세 다양성 기반 샘플링

"이미 뽑은 자세들과 충분히 달라 보일 때만" 프레임을 선택하는 알고리즘.
사전에 "몇 장 뽑을지"를 정하는 대신, **자세 차이가 임계값(threshold)을 넘을 때마다** 프레임을 채택합니다.
(k-평균처럼 개수를 고정하면, 실제로 자세 종류가 50개인데 10장만 요청 시 나머지 40개 자세가
통째로 유실되는 문제가 있음 — 이를 피하기 위해 threshold 기반 온라인 novelty 선택 방식을 사용)

### 왜 "프레임 단위"가 아니라 "돼지 개체 단위"로 비교하는가

한 프레임에 돼지가 여러 마리(군집 사육) 있기 때문에, 프레임 전체를 하나의 벡터로 보지 않고
**프레임 안의 개별 돼지 자세 벡터** 단위로 새로움(novelty)을 판단합니다.
프레임 안의 돼지 중 **단 한 마리라도** 지금까지 뽑아둔 자세 pool과 충분히 다르면
그 프레임 전체를 채택합니다 (CVAT은 프레임 단위로 라벨링하므로).
채택된 프레임에 있는 **모든** 돼지의 자세 벡터는 pool에 누적되어, 이후 프레임 판단 기준이 됩니다.

### 알고리즘 단계

1. **후보 샘플링**: 재시작 지점(영상+분)부터 끝까지, `--sample_interval_sec`(기본 2초) 간격으로 후보 프레임 목록 생성
2. **Pose 추출**: 각 후보 프레임을 돼지 전용 YOLO-pose 모델(`train4_best.pt`, 18 keypoints)에 통과시켜
   프레임 안의 모든 돼지 instance에 대해 `(18, 3)` keypoint(x, y, confidence) + bbox 획득
3. **정규화**: 각 돼지의 keypoint를 bbox 중심으로 평행이동, bbox 대각선 길이로 스케일 정규화
   → 카메라와의 거리·프레임 내 위치가 달라도 순수 "자세" 차이만 비교 가능
   (회전은 정규화하지 않음 — 돼지가 실제로 방향을 트는 것도 유의미한 자세 변화이므로 그대로 반영)
4. **가시성(occlusion) 처리**: keypoint별 confidence가 임계값(`--kpt_conf_thres`, 기본 0.3) 미만이면 "가려짐"으로 취급.
   두 자세를 비교할 때는 **양쪽 모두에서 보이는 keypoint만** 좌표 비교에 사용
5. **거리 함수** `pose_distance(A, B)`:
   - 공통으로 보이는 keypoint들의 정규화 좌표 간 평균 유클리드 거리
   - `+ occlusion_weight × (가려짐 패턴이 달라진 비율)` — 자세뿐 아니라 **가려짐 상태 자체가 달라지는 것도
     별도의 "다른 상태"로 인정** (가려짐 다양성도 라벨링 데이터에 포함하고 싶다는 요구사항 반영)
   - 공통으로 보이는 keypoint가 6개 미만이면 "비교 불가"로 처리
6. **그리디 선택**:
   ```
   pool = []  # 지금까지 채택된 모든 돼지 자세 벡터
   for 후보 프레임 (시간순):
       각 돼지 instance에 대해 pool 내 가장 가까운 자세까지의 거리 계산
       frame_novelty = 그 중 최댓값 (프레임 내 가장 새로운 돼지 기준)
       if frame_novelty > threshold or pool이 비교 불가(=완전히 새로움):
           프레임 채택, 프레임의 모든 돼지 자세를 pool에 추가
   ```
   → 자세 종류가 몇 개든 상관없이 실제로 "다른 것"만 자연스럽게 다 걸러짐.
   pool이 쌓일수록 이미 본 자세는 점점 채택되기 어려워져(포화), 반복 자세 과다 채택을 방지함.
7. 채택된 프레임을 시간 순서대로 이어 100장씩 끊어서 `segment_XXX` 폴더에 저장
   (`--start_folder`로 기존 폴더 뒤에 이어붙임)

### Threshold(임계값) 정하는 법 — `--calibrate`

자세 차이의 "적당한 크기"는 정규화 방식·keypoint 배치에 따라 달라지므로 고정 상수가 없습니다.
`--calibrate` 모드로 실제 영상 일부 구간을 스캔해서 연속 프레임 간 자세 변화량 분포를 뽑고,
percentile 기반으로 시작값을 추천받습니다.

```bash
python extract_diverse_frames.py \
    --input_dir ./day_videos --output_dir ./cvat_frames_day \
    --model_path ./models/trained/train4_best.pt \
    --resume_video "<영상 파일명 일부>" --resume_minute 18 \
    --sample_interval_sec 2 \
    --calibrate --calibrate_minutes 10 --device cuda
```

실제 day_videos 10분 구간 캘리브레이션 결과 예시:

| percentile | 값 |
|---|---|
| 50th | 0.105 |
| 70th | 0.149 |
| 80th | 0.225 |
| 90th | 0.403 |
| 95th | 0.471 |

→ 80th percentile(상위 20%만 "새 자세"로 인정)인 **0.225**를 시작 threshold로 채택.
실제 실행 후 채택되는 장수가 너무 많으면 threshold를 올리고, 너무 적으면 내려서 재조정합니다.

### 실제 추출 실행

```bash
python extract_diverse_frames.py \
    --input_dir ./day_videos \
    --output_dir ./cvat_frames_day \
    --model_path ./models/trained/train4_best.pt \
    --start_folder 3 \
    --frames_per_folder 100 \
    --resume_video "20260716012647-20260716015128" \
    --resume_minute 18 \
    --sample_interval_sec 2 \
    --pose_diff_threshold 0.225 \
    --device cuda
```

### 성능 참고

- GPU(RTX 5060) 기준 YOLO-pose 순수 추론은 프레임당 약 9ms로 매우 빠르지만,
  이 프로젝트의 AVI 파일들은 임의 위치 seek(`cap.set(CAP_PROP_POS_FRAMES, ...)`) 비용이 커서
  **seek + decode + 추론 합산 약 250~300ms/프레임**이 실제 병목입니다.
  (같은 파일에서 seek 없이 순차 `read()`만 하면 프레임당 4ms 수준으로 훨씬 빠르지만,
  이 AVI들은 seek 이후 순차 읽기가 깨지는 특성이 있어 후보 프레임마다 매번 seek이 필요함)
- 따라서 `--sample_interval_sec`을 줄이면(더 촘촘히 볼수록) 정교해지지만 선형적으로 느려짐
- 초기 구현에서는 pool과의 거리 비교를 파이썬 for문으로 pool 항목 하나하나 돌며 계산했는데,
  pool이 수만 개로 커지자 후보 하나 처리 시간이 계속 느려져 4시간 넘게 걸리고도 결과가 안 나오는
  문제가 있었음. `PoseVectorPool` 클래스로 pool을 numpy 배열에 담아 **거리 계산을 벡터화**하고,
  세그먼트가 채워질 때마다 **즉시 디스크에 저장**(끝까지 기다리지 않고 중간 결과 확인 가능)하도록
  고쳐서 실제로는 약 78분에 완료됨 (day_videos 후보 14,018개 기준)

**한계**: threshold 방식은 "몇 장이 뽑힐지"를 사전에 통제할 수 없습니다.
실제로 한 프레임에 돼지가 8~11마리씩 있다 보니(군집 사육), "그중 한 마리라도 새로운 자세면
프레임 전체 채택"하는 규칙 때문에 채택률이 예상보다 훨씬 높아짐 (day_videos 예시: 후보 14,018개
중 4,956개 채택, 약 35%). 목표 라벨링 장수가 정해져 있다면 이 후보 전체를 다 쓰기보다
3단계로 넘어가 정확한 목표 장수만 골라내는 것을 권장합니다.

## 3단계 — `select_diverse_subset.py`: FPS로 목표 장수만 재선택

2단계 결과(threshold를 넘긴 모든 후보, 장수 통제 불가)에서 **정확히 N장**만, **서로 최대한 안 겹치게**
다시 골라내는 단계. Farthest-Point Sampling(FPS) 사용.

### 알고리즘

1. 후보 이미지 각각에서 2단계와 동일한 방식으로 돼지 자세 벡터를 다시 추출
2. 두 프레임(이미지) 사이의 거리 = **Chamfer distance**
   - 프레임 A의 각 돼지가 프레임 B에서 가장 비슷한 돼지까지의 거리를 구해 평균
   - 반대 방향(B→A)도 구해서 평균 → 대칭 거리
   - 즉 "이 두 프레임에 있는 돼지들의 자세 구성이 전체적으로 얼마나 다른가"
3. **그리디 FPS**: 임의의 한 장에서 시작해, 매 반복마다 "지금까지 뽑힌 집합과 가장 먼(=가장 다른) 프레임"을
   하나씩 추가 → 목표 장수(N)에 도달할 때까지 반복
   - threshold 방식과 달리 **개수가 정확히 N장으로 고정**되면서도, 그 안에서 서로 최대한 안 겹치는
     조합을 고르므로 "고정 개수인데 다양성을 못 살리는" 문제를 피함

### 다음 배치와 이어가기 — `--save_pool` / `--load_pool`

한 번에 800~1,000장을 다 뽑는 게 아니라 영상이 추가될 때마다 여러 번에 나눠 진행하는 경우,
이번에 확정한 N장의 자세 벡터를 `--save_pool`로 저장해두면, 다음 배치에서 새 후보를 뽑을 때
`--load_pool`로 불러와서 **이전에 이미 확보한 자세와 겹치지 않는 프레임부터** 우선 선택합니다.
이렇게 하면 배치를 나눠 진행해도 전체 누적 데이터셋의 다양성이 계속 유지됩니다.

```bash
python select_diverse_subset.py \
    --input_glob "cvat_frames_day/_all_threshold_candidates/segment_0*/*.jpg" \
    --model_path ./models/trained/train4_best.pt \
    --target_count 200 \
    --output_dir ./cvat_frames_day \
    --start_folder 3 \
    --frames_per_folder 100 \
    --save_pool ./cvat_frames_day/pose_pool_state.npz \
    --device cuda
```

day_videos 실제 실행 결과: 후보 4,956장 → pose 재추출 83초 + FPS 선택 46초 (총 약 2분) →
서로 가장 다른 200장 확정 (`segment_003`, `segment_004`).

## 현재까지의 진행 상황 (day_videos 기준)

| segment | 방식 | 비고 |
|---|---|---|
| segment_001 | (프로젝트 이전 단계에서 생성된 기존 데이터, CH005 영상 90장) | 건드리지 않고 보존 |
| segment_002 | `extract_frame.py`, 1분당 1장 | CH010 영상 27개 |
| segment_003~004 | `extract_diverse_frames.py`(후보 4,956장) → `select_diverse_subset.py`(FPS)로 200장 재선택 | `[CH010] 20260716012647-20260716015128.avi`의 18번째 분부터 day_videos 끝까지 스캔한 후보 중 선정 |

현재 목표는 **총 800~1,000장**을 여러 배치에 걸쳐 채우는 것입니다. 다음 배치(새 영상 추가 시)는
`pose_pool_state.npz`를 `--load_pool`로 불러와 지금까지의 200장과 겹치지 않는 프레임부터 이어서 선택하고,
`segment_005`부터 계속 채워나가면 됩니다.

night_videos는 아직 자세 다양성 기반 추출을 적용하지 않았습니다 (`cvat_frames_night`는 전부 1단계 결과).

**백업 폴더** (참고/비교용, CVAT에는 올리지 않음):
- `cvat_frames_day/_old_1min_sampling/` — 2단계 알고리즘 적용 전, 1분 간격으로 생성했던 segment_003~007
- `cvat_frames_day/_all_threshold_candidates/` — 3단계(FPS) 적용 전, threshold만으로 거른 전체 후보 4,956장
