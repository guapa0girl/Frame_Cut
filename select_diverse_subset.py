"""
select_diverse_subset.py

extract_diverse_frames.py로 뽑아둔 "후보" 프레임들(threshold를 넘기만 하면 전부 채택되므로
장수가 많아짐) 중에서, 실제로 라벨링할 정확한 목표 장수(N장)만 "서로 가장 다르게"
다시 골라내는 스크립트.

알고리즘: Farthest-Point Sampling (FPS)
---------------------------------------
1. 각 이미지에서 돼지들의 정규화 자세 벡터를 뽑는다 (extract_diverse_frames.py와 동일 로직/정규화)
2. 두 이미지(프레임) 사이의 거리 = Chamfer distance
   - 프레임 A의 각 돼지가 프레임 B에서 가장 비슷한 돼지까지의 거리를 구해 평균
   - 반대 방향(B -> A)도 구해서 평균 (대칭 거리)
   - 즉 "이 두 프레임이 전체적으로 얼마나 다른 돼지 자세 구성을 보여주는가"
3. 그리디: 이미 뽑힌 집합과 가장 먼(가장 다른) 프레임을 하나씩 반복 추가
   -> 목표 장수(N)에 도달할 때까지 반복
   -> extract_diverse_frames.py의 threshold 방식과 달리 "정확히 N장"이 보장되고,
      개수가 고정되어 있어도 서로 최대한 겹치지 않는 조합을 고름

향후 배치 이어가기 (--load_pool)
--------------------------------
이번에 선택된 N장의 자세 벡터를 --save_pool 로 저장해두면, 나중에 새 영상에서 후보를
추가로 뽑았을 때 --load_pool 로 이 상태를 불러와 "이미 뽑은 자세와 안 겹치는" 새 프레임을
이어서 고를 수 있다 (총 800~1000장까지 여러 번에 걸쳐 확장하는 시나리오에 대응).

사용법
------
python select_diverse_subset.py \
    --input_glob "cvat_frames_day/segment_0[0-5][0-9]/*.jpg" \
    --model_path ./models/trained/train4_best.pt \
    --target_count 200 \
    --output_dir ./cvat_frames_day \
    --start_folder 3 \
    --frames_per_folder 100 \
    --save_pool ./cvat_frames_day/pose_pool_state.npz \
    --device cuda
"""

import argparse
import glob
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from extract_diverse_frames import NUM_KPTS, extract_pose_vectors

HIGH_DIST = 3.0  # 비교 불가/전혀 안 겹치는 경우에 부여하는 큰 거리 값


def load_frame_poses(model, image_paths, det_conf_thres, kpt_conf_thres, device):
    frame_poses = []
    t0 = time.time()
    for i, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            frame_poses.append([])
            continue
        r = model.predict(img, verbose=False, device=device)[0]
        poses = extract_pose_vectors(r, det_conf_thres, kpt_conf_thres)
        frame_poses.append(poses)
        if (i + 1) % 500 == 0:
            print(f"  [pose 추출] {i+1}/{len(image_paths)}  경과 {time.time()-t0:.0f}s", flush=True)
    return frame_poses


def pair_dist_matrix(A, B, occlusion_weight, min_shared_kpts):
    """A(nA개 자세), B(nB개 자세) -> (nA,nB) 거리 행렬."""
    if not A or not B:
        return np.full((max(len(A), 1), max(len(B), 1)), HIGH_DIST)
    Axy = np.stack([a["xy"] for a in A])
    Avis = np.stack([a["vis"] for a in A])
    Bxy = np.stack([b["xy"] for b in B])
    Bvis = np.stack([b["vis"] for b in B])

    diff = Axy[:, None, :, :] - Bxy[None, :, :, :]              # (nA,nB,18,2)
    per_kpt = np.sqrt((diff ** 2).sum(axis=3))                   # (nA,nB,18)
    shared = Avis[:, None, :] & Bvis[None, :, :]                  # (nA,nB,18)
    shared_count = shared.sum(axis=2)                             # (nA,nB)
    sum_dist = np.where(shared, per_kpt, 0.0).sum(axis=2)
    kpt_dist = sum_dist / np.maximum(shared_count, 1)
    occ_mismatch = (Avis[:, None, :] != Bvis[None, :, :]).mean(axis=2)
    dist = kpt_dist + occlusion_weight * occ_mismatch
    dist = np.where(shared_count >= min_shared_kpts, dist, HIGH_DIST)
    return dist


def chamfer_distance(A, B, occlusion_weight, min_shared_kpts):
    if not A or not B:
        return HIGH_DIST
    D = pair_dist_matrix(A, B, occlusion_weight, min_shared_kpts)
    a_to_b = D.min(axis=1).mean()
    b_to_a = D.min(axis=0).mean()
    return float(0.5 * (a_to_b + b_to_a))


def farthest_point_sampling(frame_poses, target_count, occlusion_weight, min_shared_kpts,
                             seed_min_dist=None):
    """seed_min_dist: 이전에 저장된 pool과의 거리로 초기화하고 싶을 때 (N,) 배열."""
    n = len(frame_poses)
    target_count = min(target_count, n)

    if seed_min_dist is not None:
        min_dist = seed_min_dist.copy()
        selected = []
        remaining = set(range(n))
    else:
        min_dist = np.full(n, np.inf)
        seed = 0
        selected = [seed]
        remaining = set(range(n))
        remaining.discard(seed)
        for j in remaining:
            min_dist[j] = chamfer_distance(frame_poses[seed], frame_poses[j],
                                            occlusion_weight, min_shared_kpts)

    t0 = time.time()
    while len(selected) < target_count and remaining:
        best_j = max(remaining, key=lambda j: min_dist[j])
        selected.append(best_j)
        remaining.discard(best_j)
        for j in remaining:
            d = chamfer_distance(frame_poses[best_j], frame_poses[j], occlusion_weight, min_shared_kpts)
            if d < min_dist[j]:
                min_dist[j] = d
        if len(selected) % 20 == 0:
            print(f"  [FPS] {len(selected)}/{target_count} 선택, 경과 {time.time()-t0:.0f}s", flush=True)

    return selected


def main():
    parser = argparse.ArgumentParser(description="후보 프레임 중 서로 가장 다른 N장만 FPS로 재선택")
    parser.add_argument("--input_glob", required=True, help="후보 이미지 glob 패턴")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target_count", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_folder", type=int, default=1)
    parser.add_argument("--frames_per_folder", type=int, default=100)
    parser.add_argument("--occlusion_weight", type=float, default=0.5)
    parser.add_argument("--det_conf_thres", type=float, default=0.5)
    parser.add_argument("--kpt_conf_thres", type=float, default=0.3)
    parser.add_argument("--min_shared_kpts", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_pool", default=None, help="선택된 프레임들의 자세 벡터 저장 경로 (.npz)")
    parser.add_argument("--load_pool", default=None, help="이전에 저장된 pool(.npz)을 불러와 이어서 선택")
    args = parser.parse_args()

    image_paths = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not image_paths:
        print(f"[ERROR] 후보 이미지를 찾을 수 없음: {args.input_glob}")
        return
    print(f"[INFO] 후보 {len(image_paths)}장, 목표 {args.target_count}장 선택 예정", flush=True)

    from ultralytics import YOLO
    model = YOLO(args.model_path)
    model.to(args.device)

    print("[INFO] 후보 프레임 pose 추출 중...", flush=True)
    frame_poses = load_frame_poses(model, image_paths, args.det_conf_thres, args.kpt_conf_thres, args.device)

    seed_min_dist = None
    if args.load_pool:
        pool = np.load(args.load_pool, allow_pickle=True)
        pool_xy, pool_vis = pool["xy"], pool["vis"]
        pool_poses = [{"xy": pool_xy[i], "vis": pool_vis[i]} for i in range(len(pool_xy))]
        # pool 자체는 여러 프레임의 돼지들을 한 줄로 모아둔 것이므로, 프레임 단위 pool로 재구성
        # (프레임 경계 정보가 없으므로 전체를 하나의 "가상 프레임"으로 취급해 비교)
        print(f"[INFO] 이전 pool 로드: {len(pool_poses)}개 자세 벡터. 여기와 안 겹치는 프레임부터 선택합니다.", flush=True)
        seed_min_dist = np.array([
            chamfer_distance(fp, pool_poses, args.occlusion_weight, args.min_shared_kpts)
            for fp in frame_poses
        ])

    print("[INFO] Farthest-Point Sampling 진행 중...", flush=True)
    selected_idx = farthest_point_sampling(frame_poses, args.target_count,
                                            args.occlusion_weight, args.min_shared_kpts,
                                            seed_min_dist=seed_min_dist)

    output_dir = Path(args.output_dir)
    start_offset = args.start_folder - 1
    total = len(selected_idx)
    num_folders = max(1, -(-total // args.frames_per_folder))
    print(f"\n[INFO] 총 {total}장 선택 -> {num_folders}개 폴더 (segment_{args.start_folder:03d} 부터)", flush=True)

    for local_global_idx, src_idx in enumerate(selected_idx):
        folder_idx = local_global_idx // args.frames_per_folder + start_offset
        local_idx = local_global_idx % args.frames_per_folder
        out_dir = output_dir / f"segment_{folder_idx + 1:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        src = image_paths[src_idx]
        dst = out_dir / f"frame_{local_idx + 1:04d}__{src.stem}.{src.suffix.lstrip('.')}"
        shutil.copy2(src, dst)

    print("\n[DONE]")
    for folder_idx in range(start_offset, start_offset + num_folders):
        d = output_dir / f"segment_{folder_idx + 1:03d}"
        count = len(list(d.glob("*.jpg"))) if d.exists() else 0
        print(f"  segment_{folder_idx + 1:03d}: {count}장 -> {d}")

    if args.save_pool:
        all_xy = np.concatenate([np.stack([p["xy"] for p in frame_poses[i]])
                                  for i in selected_idx if frame_poses[i]], axis=0)
        all_vis = np.concatenate([np.stack([p["vis"] for p in frame_poses[i]])
                                   for i in selected_idx if frame_poses[i]], axis=0)
        np.savez(args.save_pool, xy=all_xy, vis=all_vis)
        print(f"\n[INFO] 선택된 {total}장의 자세 pool 저장: {args.save_pool} (자세 벡터 {len(all_xy)}개)")
        print("       -> 다음 배치 추출 시 --load_pool 로 불러오면 이 프레임들과 겹치지 않는 것부터 고릅니다.")


if __name__ == "__main__":
    main()
