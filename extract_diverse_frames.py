"""
extract_diverse_frames.py

돼지 pose estimation 모델(YOLO-pose, 18 keypoints)을 이용해
"자세가 확실히 달라질 때만" 프레임을 뽑는 스크립트.

extract_frame.py(1분당 1장 고정 샘플링)의 후속 단계로, segment_003부터는
시간 간격이 아니라 "자세 다양성"을 기준으로 프레임을 선택한다.

핵심 아이디어
------------
- k개를 미리 정해서 뽑는 게 아니라, "이미 뽑아둔 자세들과 충분히 다른 자세가
  나타날 때마다" 뽑는다 (threshold 기반 온라인 novelty 선택).
- 한 프레임에 돼지가 여러 마리(그룹 사육) 있으므로, 프레임 단위가 아니라
  "프레임 안의 개별 돼지 자세 벡터" 단위로 novelty를 계산한다.
  프레임 안의 돼지 중 단 한 마리라도 지금까지 뽑은 자세 풀(pool)과
  충분히 다르면 그 프레임 전체를 선택한다 (CVAT은 프레임 단위 라벨링이므로).
- 선택된 프레임의 모든 돼지 자세 벡터는 pool에 누적되어, 이후 프레임의
  novelty 판단 기준이 된다 (이미 본 자세는 계속 걸러짐).

자세 벡터
--------
- YOLO-pose로 얻은 (18, 3) keypoint (x, y, conf)를 bbox 중심으로 이동,
  bbox 대각선 길이로 스케일 정규화 -> 위치/크기 불변 벡터
- keypoint별 conf가 낮으면(=가려짐) 그 keypoint는 두 자세를 비교할 때 제외
- 두 자세 사이 거리 = (공통으로 보이는 keypoint들의 정규화 좌표 거리) 가중평균
  + (보임/가려짐 패턴이 달라진 정도) 가중치 합
  -> 자세 자체가 달라지는 경우뿐 아니라, 가려짐 상태가 달라지는 경우도 포착

사용법
------
1) 캘리브레이션 (임계값 감 잡기, 저장 없음):
   python extract_diverse_frames.py --calibrate --calibrate_minutes 10 ...

2) 실제 추출:
   python extract_diverse_frames.py \
       --input_dir ./day_videos \
       --output_dir ./cvat_frames_day \
       --model_path ./models/trained/train4_best.pt \
       --start_folder 3 \
       --frames_per_folder 100 \
       --resume_video "20260716012647-20260716015128" \
       --resume_minute 18 \
       --sample_interval_sec 2 \
       --pose_diff_threshold 0.35
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
NUM_KPTS = 18


def find_videos(input_dir: Path, recursive: bool):
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def read_frame_safe(cap, target_idx: int, max_backoff: int = 30):
    for offset in range(0, max_backoff + 1):
        idx = target_idx - offset
        if idx < 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, idx, frame
    return False, target_idx, None


def get_video_meta(video_path: Path, expected_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = expected_fps
    return fps, total_frames


def extract_pose_vectors(result, det_conf_thres: float, kpt_conf_thres: float):
    """YOLO pose 결과 하나(=한 프레임)에서 각 돼지 instance의 정규화된 자세 벡터를 뽑는다.
    반환: list of dict(xy: (18,2) normalized, vis: (18,) bool, conf: float, bbox: (4,))
    """
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    kpts = result.keypoints.data.detach().cpu().numpy()  # (N, 18, 3)

    for i in range(len(boxes)):
        if confs[i] < det_conf_thres:
            continue
        x1, y1, x2, y2 = boxes[i]
        w, h = x2 - x1, y2 - y1
        scale = max(float(np.hypot(w, h)), 1e-3)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        kp = kpts[i]  # (18, 3)
        xy = kp[:, :2].astype(np.float32)
        kconf = kp[:, 2].astype(np.float32)

        norm_xy = np.zeros((NUM_KPTS, 2), dtype=np.float32)
        norm_xy[:, 0] = (xy[:, 0] - cx) / scale
        norm_xy[:, 1] = (xy[:, 1] - cy) / scale
        vis = kconf >= kpt_conf_thres

        out.append({
            "xy": norm_xy,
            "vis": vis,
            "conf": float(confs[i]),
        })
    return out


class PoseVectorPool:
    """지금까지 채택된 모든 돼지 자세 벡터를 numpy 배열로 보관 (동적 배열, 2배씩 증설).
    pool이 수천 개로 커져도, 후보 하나당 거리 계산을 파이썬 루프 대신 한 번의
    벡터화된 numpy 연산으로 처리해서 O(pool 크기)의 상수 비용을 크게 줄인다."""

    def __init__(self, min_shared_kpts: int = 6, occlusion_weight: float = 0.5, capacity: int = 1024):
        self.min_shared_kpts = min_shared_kpts
        self.occlusion_weight = occlusion_weight
        self.n = 0
        self.capacity = capacity
        self.xy = np.zeros((capacity, NUM_KPTS, 2), dtype=np.float32)
        self.vis = np.zeros((capacity, NUM_KPTS), dtype=bool)

    def _grow(self, min_capacity: int):
        new_capacity = max(self.capacity * 2, min_capacity)
        new_xy = np.zeros((new_capacity, NUM_KPTS, 2), dtype=np.float32)
        new_vis = np.zeros((new_capacity, NUM_KPTS), dtype=bool)
        new_xy[: self.n] = self.xy[: self.n]
        new_vis[: self.n] = self.vis[: self.n]
        self.xy, self.vis = new_xy, new_vis
        self.capacity = new_capacity

    def add(self, poses):
        if not poses:
            return
        k = len(poses)
        if self.n + k > self.capacity:
            self._grow(self.n + k)
        for p in poses:
            self.xy[self.n] = p["xy"]
            self.vis[self.n] = p["vis"]
            self.n += 1

    def min_dist(self, pose):
        """pool이 비어있거나 비교 가능한 항목이 하나도 없으면 '매우 새로움'(inf)."""
        if self.n == 0:
            return float("inf")
        pool_xy = self.xy[: self.n]       # (P, 18, 2)
        pool_vis = self.vis[: self.n]     # (P, 18)
        shared = pool_vis & pose["vis"][None, :]        # (P, 18)
        shared_count = shared.sum(axis=1)                # (P,)
        valid = shared_count >= self.min_shared_kpts
        if not valid.any():
            return float("inf")

        diff = pool_xy - pose["xy"][None, :, :]           # (P, 18, 2)
        per_kpt_dist = np.sqrt((diff ** 2).sum(axis=2))   # (P, 18)
        sum_dist = np.where(shared, per_kpt_dist, 0.0).sum(axis=1)  # (P,)
        kpt_dist = sum_dist / np.maximum(shared_count, 1)           # (P,)

        occ_mismatch = (pool_vis != pose["vis"][None, :]).mean(axis=1)  # (P,)
        dists = kpt_dist + self.occlusion_weight * occ_mismatch

        return float(dists[valid].min())


def build_candidate_plan(videos, resume_video_substr, resume_minute, sample_interval_sec, expected_fps):
    """resume 지점부터 끝까지, sample_interval_sec 간격의 (video_path, frame_idx) 후보 목록 생성."""
    resume_idx = None
    for i, v in enumerate(videos):
        if resume_video_substr in v.name:
            resume_idx = i
            break
    if resume_idx is None:
        print(f"[ERROR] resume_video 문자열을 포함한 영상을 찾을 수 없음: {resume_video_substr}")
        sys.exit(1)

    plan = []
    for i in range(resume_idx, len(videos)):
        video_path = videos[i]
        meta = get_video_meta(video_path, expected_fps)
        if meta is None:
            print(f"[ERROR] 열 수 없음: {video_path}")
            continue
        fps, total_frames = meta
        step = max(1, int(round(fps * sample_interval_sec)))

        if i == resume_idx and resume_minute is not None:
            minute_step = int(round(fps * 60))
            start_frame = (resume_minute - 1) * minute_step
        else:
            start_frame = 0

        for frame_idx in range(start_frame, total_frames, step):
            plan.append((video_path, frame_idx))

    return plan


def run_calibration(args, model):
    import cv2
    videos = find_videos(Path(args.input_dir), args.recursive)
    plan = build_candidate_plan(videos, args.resume_video, args.resume_minute,
                                 args.sample_interval_sec, args.expected_fps)
    n = min(len(plan), int(args.calibrate_minutes * 60 / args.sample_interval_sec))
    plan = plan[:n]
    print(f"[CALIBRATE] 후보 {len(plan)}개로 프레임 간 자세 변화량 분포 측정 중...")

    prev_pool = None
    scores = []
    cur_video = None
    cap = None
    t0 = time.time()
    for k, (video_path, frame_idx) in enumerate(plan):
        if video_path != cur_video:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(video_path))
            cur_video = video_path
        ok, actual_idx, frame = read_frame_safe(cap, frame_idx)
        if not ok:
            continue
        r = model.predict(frame, verbose=False, device=args.device)[0]
        poses = extract_pose_vectors(r, args.det_conf_thres, args.kpt_conf_thres)
        if prev_pool is not None:
            frame_score = 0.0
            for p in poses:
                d = prev_pool.min_dist(p)
                if d != float("inf"):
                    frame_score = max(frame_score, d)
            if frame_score > 0:
                scores.append(frame_score)
        if poses:
            prev_pool = PoseVectorPool(occlusion_weight=args.occlusion_weight, capacity=max(len(poses), 16))
            prev_pool.add(poses)
        if (k + 1) % 50 == 0:
            print(f"  ...{k+1}/{len(plan)} 처리, 경과 {time.time()-t0:.0f}s", flush=True)
    if cap is not None:
        cap.release()

    if not scores:
        print("[CALIBRATE] 유효한 점수를 얻지 못함 (감지 실패 등)")
        return

    scores = np.array(scores)
    print(f"\n[CALIBRATE] n={len(scores)}  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}  std={scores.std():.4f}")
    for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
        print(f"  {pct}th percentile: {np.percentile(scores, pct):.4f}")
    suggested = np.percentile(scores, 80)
    print(f"\n[CALIBRATE] 추천 --pose_diff_threshold 시작값(80th pct): {suggested:.4f}")
    print("[CALIBRATE] 실제 실행 시 뽑히는 장수를 보고 필요하면 올리거나(더 적게) 내려서(더 많이) 조정하세요.")


def main():
    parser = argparse.ArgumentParser(description="자세 다양성 기반 프레임 추출 (YOLO-pose, threshold 방식)")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--frames_per_folder", type=int, default=100)
    parser.add_argument("--start_folder", type=int, default=1,
                         help="segment 폴더 시작 번호 (기존 앞 번호 폴더는 건드리지 않음)")
    parser.add_argument("--resume_video", required=True,
                         help="이 문자열을 파일명에 포함하는 영상부터 스캔 시작")
    parser.add_argument("--resume_minute", type=int, default=1,
                         help="resume_video 안에서 몇 번째 분(1분 단위, min001 기준)부터 시작할지")
    parser.add_argument("--sample_interval_sec", type=float, default=2.0,
                         help="후보 프레임을 몇 초 간격으로 볼지 (촘촘할수록 느리지만 정교함)")
    parser.add_argument("--pose_diff_threshold", type=float, default=0.35,
                         help="이 값보다 자세 차이가 크면 새 자세로 인정 (--calibrate로 추천값 확인)")
    parser.add_argument("--occlusion_weight", type=float, default=0.5,
                         help="가려짐 패턴 변화에 부여할 가중치")
    parser.add_argument("--det_conf_thres", type=float, default=0.5, help="돼지 검출 최소 confidence")
    parser.add_argument("--kpt_conf_thres", type=float, default=0.3, help="keypoint '보임' 판정 최소 confidence")
    parser.add_argument("--expected_fps", type=float, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ext", default="jpg")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="저장 없이 임계값 추천만 출력")
    parser.add_argument("--calibrate_minutes", type=float, default=10.0)
    parser.add_argument("--max_frames", type=int, default=None,
                         help="테스트용: 최대 이만큼만 선택하고 종료")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model_path)
    model.to(args.device)

    if args.calibrate:
        run_calibration(args, model)
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    videos = find_videos(input_dir, args.recursive)
    if not videos:
        print(f"[ERROR] {input_dir} 에서 영상을 찾을 수 없음")
        sys.exit(1)

    plan = build_candidate_plan(videos, args.resume_video, args.resume_minute,
                                 args.sample_interval_sec, args.expected_fps)
    print(f"[INFO] resume={args.resume_video} min{args.resume_minute:03d} 부터 후보 {len(plan)}개 스캔 예정 "
          f"({args.sample_interval_sec}초 간격)", flush=True)

    pool = PoseVectorPool(occlusion_weight=args.occlusion_weight, capacity=2048)
    start_offset = args.start_folder - 1
    total_selected = 0
    cur_video = None
    cap = None
    t0 = time.time()

    def make_folder(folder_idx):
        d = output_dir / f"segment_{folder_idx + 1:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    make_folder(start_offset)  # 첫 세그먼트 폴더 미리 생성

    for k, (video_path, frame_idx) in enumerate(plan):
        if video_path != cur_video:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(video_path))
            cur_video = video_path
            print(f"[INFO] {video_path.name} 스캔 중... (지금까지 선택 {total_selected}장)", flush=True)

        ok, actual_idx, frame = read_frame_safe(cap, frame_idx)
        if not ok:
            continue

        r = model.predict(frame, verbose=False, device=args.device)[0]
        poses = extract_pose_vectors(r, args.det_conf_thres, args.kpt_conf_thres)
        if not poses:
            continue

        frame_novelty = 0.0
        for p in poses:
            d = pool.min_dist(p)
            if d == float("inf"):
                frame_novelty = float("inf")
                break
            frame_novelty = max(frame_novelty, d)

        if frame_novelty > args.pose_diff_threshold or frame_novelty == float("inf"):
            # 채택 -> 즉시 디스크에 저장 (segment 100장 단위)
            folder_idx = total_selected // args.frames_per_folder + start_offset
            local_idx = total_selected % args.frames_per_folder
            if local_idx == 0:
                make_folder(folder_idx)
            nov_str = "inf" if frame_novelty == float("inf") else f"{frame_novelty:.3f}"
            label = f"{video_path.stem}_f{actual_idx:07d}_nov{nov_str}"
            out_path = make_folder(folder_idx) / f"frame_{local_idx + 1:04d}__{label}.{args.ext}"
            cv2.imwrite(str(out_path), frame)

            pool.add(poses)
            total_selected += 1

        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{k+1}/{len(plan)}] 경과 {elapsed/60:.1f}분, 선택 {total_selected}장, pool {pool.n}개",
                  flush=True)

        if args.max_frames and total_selected >= args.max_frames:
            print(f"[INFO] max_frames={args.max_frames} 도달, 중단", flush=True)
            break

    if cap is not None:
        cap.release()

    num_folders = max(1, -(-total_selected // args.frames_per_folder))
    print(f"\n[DONE] 총 선택 {total_selected}장 -> {num_folders}개 폴더 (segment_{args.start_folder:03d} 부터)")
    for folder_idx in range(start_offset, start_offset + num_folders):
        d = output_dir / f"segment_{folder_idx + 1:03d}"
        count = len(list(d.glob(f"*.{args.ext}"))) if d.exists() else 0
        print(f"  segment_{folder_idx + 1:03d}: {count}장 -> {d}")


if __name__ == "__main__":
    main()
