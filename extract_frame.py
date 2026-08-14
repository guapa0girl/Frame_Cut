"""
extract_frames_for_cvat.py

돼지 pose estimation 라벨링(CVAT)을 위한 프레임 추출 스크립트.

동작 방식
---------
1. 지정한 폴더 안의 모든 동영상을 찾는다.
2. 각 영상에서 "1분에 1장씩" 프레임을 추출한다.
   (예: 138분짜리 영상 -> 138장, 20분짜리 영상 -> 20장)
3. 모든 영상에서 뽑힌 프레임을 하나로 합친 뒤(영상 순서 -> 그 안에서 시간 순서),
   100장씩 순서대로 끊어서 폴더에 저장한다. (영상별 폴더 구분 없음)
   - segment_001 : 1~100번째 프레임
   - segment_002 : 101~200번째 프레임
   - ...

결과 구조 (폴더 하나 = CVAT task 하나로 올리기 좋음)
------
   output_root/
     ├── segment_001/
     │     ├── frame_0001__영상A_min001.jpg
     │     ├── frame_0002__영상A_min002.jpg
     │     ├── ...
     │     └── frame_0100__영상B_min015.jpg   <- 영상이 바뀌어도 한 폴더 안에 섞여 들어감
     ├── segment_002/
     │     └── ...
     └── ...

파일명에 원본 영상 이름과 몇 번째 분(minute)인지 표시해서, 폴더를 섞어 담아도
어떤 프레임이 어느 영상 것인지 나중에 확인할 수 있게 했습니다.

사용법
------
python extract_frame.py \
    --input_dir . \
    --output_dir ./cvat_frames \
    --interval_sec 60 \
    --frames_per_folder 100

옵션 설명
--------
--input_dir           영상들이 들어있는 폴더
--output_dir          결과 프레임을 저장할 폴더
--interval_sec        몇 초마다 1장 뽑을지. 기본 60초(1분에 1장)
--frames_per_folder   한 폴더(segment)에 몇 장씩 담을지. 기본 100
--ext                 저장할 이미지 확장자 (jpg 기본, png도 가능)
--expected_fps        예상 fps (메타데이터를 못 읽을 때 대체값)
--recursive           하위 폴더까지 영상 탐색
--dry_run             실제로 저장하지 않고 몇 장/몇 세그먼트가 생성될지만 미리 확인
"""

import argparse
import sys
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def find_videos(input_dir: Path, recursive: bool):
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def read_frame_safe(cap, target_idx: int, max_backoff: int = 30):
    """target_idx 프레임을 읽되, 실패하면 앞쪽(과거) 프레임으로 조금씩 물러나며 재시도.
    (AVI 등에서 메타데이터상 프레임 수와 실제 디코딩 가능한 프레임 수가 달라
    맨 끝부분 프레임 읽기가 실패하는 경우를 보정하기 위함)
    반환값: (성공 여부, 실제로 읽은 frame_idx, frame 이미지)
    """
    for offset in range(0, max_backoff + 1):
        idx = target_idx - offset
        if idx < 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, idx, frame
    return False, target_idx, None


def get_video_info(video_path: Path, interval_sec: int, expected_fps: float):
    """영상 하나를 열어 fps/총프레임을 읽고, 추출할 프레임 인덱스 목록을 반환."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 열 수 없음: {video_path}")
        return None, [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        print(f"[WARN] {video_path.name}: fps를 읽지 못해 예상값({expected_fps})을 사용합니다.")
        fps = expected_fps
    elif expected_fps and abs(fps - expected_fps) > 1:
        print(f"[WARN] {video_path.name}: 실제 fps={fps:.2f} 가 예상 fps={expected_fps}와 다릅니다. "
              f"실제 fps 기준으로 계산합니다.")

    step = int(round(fps * interval_sec))
    if step <= 0:
        print(f"[ERROR] {video_path.name}: 유효하지 않은 step({step}), 건너뜀")
        return fps, [], total_frames

    frame_indices = list(range(0, total_frames, step))
    return fps, frame_indices, total_frames


def main():
    parser = argparse.ArgumentParser(
        description="CVAT 라벨링용 프레임 추출 (모든 영상에서 N초마다 1장 -> 전체를 100장씩 폴더로 묶기)")
    parser.add_argument("--input_dir", required=True, help="영상 파일들이 있는 폴더")
    parser.add_argument("--output_dir", required=True, help="추출된 프레임을 저장할 폴더")
    parser.add_argument("--interval_sec", type=int, default=60,
                         help="몇 초마다 1장 뽑을지, 기본 60초(1분에 1장)")
    parser.add_argument("--frames_per_folder", type=int, default=100,
                         help="한 폴더에 몇 장씩 담을지, 기본 100")
    parser.add_argument("--ext", default="jpg", help="저장 이미지 확장자 (jpg, png 등)")
    parser.add_argument("--expected_fps", type=float, default=200,
                         help="예상 fps (메타데이터를 못 읽을 때 대체값)")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더까지 영상 탐색")
    parser.add_argument("--dry_run", action="store_true", help="실제 저장 없이 계획만 출력")
    parser.add_argument("--start_folder", type=int, default=1,
                         help="segment 폴더 번호를 몇 번부터 시작할지 (1부터 시작). "
                              "예: 2를 주면 segment_002, segment_003 ... 순으로 생성 "
                              "(기존 segment_001 등 앞 번호 폴더는 건드리지 않음)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] 입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)

    videos = find_videos(input_dir, args.recursive)
    if not videos:
        print(f"[ERROR] {input_dir} 에서 영상 파일을 찾을 수 없습니다. "
              f"지원 확장자: {sorted(VIDEO_EXTS)}")
        sys.exit(1)

    print(f"[INFO] 총 {len(videos)}개 영상 발견")

    # 1단계: 모든 영상을 순서대로 훑으면서 "몇 번째 프레임을 뽑을지" 목록을 전부 만든다.
    # 각 항목 = (video_path, frame_idx_in_video, source_label)
    job_list = []
    for video_path in videos:
        fps, frame_indices, total_frames = get_video_info(video_path, args.interval_sec, args.expected_fps)
        if not frame_indices:
            continue
        print(f"[INFO] {video_path.name} | fps={fps:.2f} | 총 프레임={total_frames} "
              f"| {args.interval_sec}초 간격 -> {len(frame_indices)}장 추출 예정")
        for minute_idx, frame_idx in enumerate(frame_indices, start=1):
            label = f"{video_path.stem}_min{minute_idx:03d}"
            job_list.append((video_path, frame_idx, label))

    total_extracted = len(job_list)
    num_folders = max(1, -(-total_extracted // args.frames_per_folder))  # ceil division
    start_offset = args.start_folder - 1
    print(f"\n[INFO] 전체 영상 합산 {total_extracted}장 -> {args.frames_per_folder}장씩 -> {num_folders}개 폴더 생성 예정 "
          f"(segment_{args.start_folder:03d} 부터 시작)")

    # 2단계: 전체 목록에서 global index 기준으로 몇 번 폴더(segment)에 들어갈지 미리 계산
    # start_folder로 지정한 번호만큼 폴더 번호를 밀어서 배정 (앞 번호 폴더는 건드리지 않음)
    for global_idx, (video_path, frame_idx, label) in enumerate(job_list):
        folder_idx = global_idx // args.frames_per_folder + start_offset
        local_idx = global_idx % args.frames_per_folder
        job_list[global_idx] = (video_path, frame_idx, label, folder_idx, local_idx)

    if args.dry_run:
        for folder_idx in range(start_offset, start_offset + num_folders):
            count = sum(1 for j in job_list if j[3] == folder_idx)
            print(f"  segment_{folder_idx + 1:03d}: {count}장 예정 -> {output_dir / f'segment_{folder_idx + 1:03d}'}")
        print("\n[DRY RUN] 실제 저장은 수행하지 않았습니다.")
        return

    # 폴더 미리 생성
    for folder_idx in range(start_offset, start_offset + num_folders):
        (output_dir / f"segment_{folder_idx + 1:03d}").mkdir(parents=True, exist_ok=True)

    # 3단계: 영상별로 한 번씩만 열어서 필요한 프레임들을 읽고, 정해진 폴더에 저장
    videos_in_jobs = sorted(set(j[0] for j in job_list), key=lambda p: videos.index(p))
    for video_path in videos_in_jobs:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] 열 수 없음: {video_path}")
            continue

        jobs_for_video = [j for j in job_list if j[0] == video_path]
        for (_, frame_idx, label, folder_idx, local_idx) in jobs_for_video:
            ok, actual_idx, frame = read_frame_safe(cap, frame_idx, max_backoff=30)
            if not ok:
                print(f"    [WARN] {video_path.name} 프레임 {frame_idx} 읽기 실패 "
                      f"(주변 30프레임 재시도도 실패), 건너뜀")
                continue
            if actual_idx != frame_idx:
                print(f"    [INFO] {video_path.name} 프레임 {frame_idx} 대신 {actual_idx} 사용 "
                      f"({frame_idx - actual_idx}프레임 보정)")
            out_dir = output_dir / f"segment_{folder_idx + 1:03d}"
            out_path = out_dir / f"frame_{local_idx + 1:04d}__{label}.{args.ext}"
            cv2.imwrite(str(out_path), frame)

        cap.release()
        print(f"[INFO] {video_path.name} 처리 완료 ({len(jobs_for_video)}장 저장)")

    print("\n[DONE] 모든 영상 처리 완료")
    for folder_idx in range(start_offset, start_offset + num_folders):
        count = sum(1 for j in job_list if j[3] == folder_idx)
        print(f"  segment_{folder_idx + 1:03d}: {count}장 -> {output_dir / f'segment_{folder_idx + 1:03d}'}")


if __name__ == "__main__":
    main()