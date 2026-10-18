#!/usr/bin/env python3
"""
盲道检测模型性能基准测试脚本
用法：
    python3 benchmark.py /path/to/model.cvimodel
示例：
    python3 benchmark.py /root/.picoclaw/workspace/skills/blind-lane-assistant/yolo26n_seg_bf16.cvimodel
    python3 benchmark.py /root/.picoclaw/workspace/skills/blind-lane-assistant/yolo26n_seg_int8.cvimodel
"""

import sys
import os
import time

# ============================================================
# 导入 MaixPy 和 YOLO26Seg
# ============================================================
from maix import camera, image, app, time as maix_time

SKILL_DIR = "/root/.picoclaw/workspace/skills/blind-lane-assistant/"
sys.path.insert(0, SKILL_DIR)
from yolo26_seg_maixpy import YOLO26Seg

# ============================================================
# 测试参数
# ============================================================
WARMUP_FRAMES = 10
TEST_FRAMES = 100
CONF_TH = 0.50
IOU_TH = 0.50


def run_benchmark(model_path):
    model_label = os.path.basename(model_path)

    print("=" * 60)
    print("  盲道检测模型性能基准测试")
    print("=" * 60)
    print(f"  模型文件 : {model_path}")
    print(f"  预热帧数 : {WARMUP_FRAMES}")
    print(f"  测试帧数 : {TEST_FRAMES}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[INFO] 模型文件大小: {file_size_mb:.2f} MB")

    # 加载模型
    print("[INFO] 正在加载模型...")
    load_start = maix_time.ticks_ms()
    detector = YOLO26Seg(model_path, default_input_w=640, default_input_h=640)
    detector.fast_single_class = True
    detector.max_candidates_before_nms = 80
    detector.max_results_after_nms = 1
    detector.enable_sparse_overlay = False
    load_end = maix_time.ticks_ms()
    load_time = load_end - load_start
    print(f"[INFO] 模型加载耗时: {load_time} ms")

    # 用模型实际输入尺寸初始化摄像头
    cam_w = detector.input_width()
    cam_h = detector.input_height()
    try:
        cam_fmt = image.Format.FMT_RGB888
    except AttributeError:
        cam_fmt = image.FMT_RGB888

    cam = camera.Camera(cam_w, cam_h, cam_fmt)
    print(f"[INFO] 摄像头已初始化: {cam_w}x{cam_h}")

    # ---- 预热 ----
    print(f"\n[INFO] 预热中（{WARMUP_FRAMES} 帧）...")
    for i in range(WARMUP_FRAMES):
        img = cam.read()
        if img is None:
            continue
        detector.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
        if app.need_exit():
            print("[WARN] 用户中断")
            return
    print("[INFO] 预热完成")

    # ---- 正式测试 ----
    print(f"\n[INFO] 开始正式测试（{TEST_FRAMES} 帧）...")

    frame_times = []
    infer_times = []
    capture_times = []

    for i in range(TEST_FRAMES):
        if app.need_exit():
            print("[WARN] 用户中断")
            break

        # 取帧
        t0 = maix_time.ticks_ms()
        img = cam.read()
        t1 = maix_time.ticks_ms()
        capture_times.append(t1 - t0)

        if img is None:
            continue

        # 推理（含前处理+TPU+后处理）
        objs = detector.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
        t2 = maix_time.ticks_ms()
        infer_times.append(t2 - t1)

        frame_times.append(t2 - t0)

        # 进度
        if (i + 1) % 20 == 0:
            avg_so_far = sum(frame_times) / len(frame_times)
            fps_so_far = 1000.0 / avg_so_far if avg_so_far > 0 else 0
            print(f"  [{i + 1}/{TEST_FRAMES}] 当前平均: {avg_so_far:.1f} ms/帧 ({fps_so_far:.1f} fps)")

    # ---- 统计 ----
    n = len(frame_times)
    if n == 0:
        print("[ERROR] 没有有效测试帧")
        return

    total_ms = sum(frame_times)
    avg_frame = total_ms / n
    avg_infer = sum(infer_times) / n
    avg_capture = sum(capture_times) / n
    fps = 1000.0 / avg_frame

    sorted_f = sorted(frame_times)
    median_f = sorted_f[n // 2]
    p95_f = sorted_f[min(int(n * 0.95), n - 1)]
    p99_f = sorted_f[min(int(n * 0.99), n - 1)]

    sorted_i = sorted(infer_times)
    median_i = sorted_i[n // 2]

    # ---- 输出结果 ----
    print()
    print("=" * 60)
    print(f"  测试结果: {model_label}")
    print("=" * 60)
    print(f"  模型文件大小          : {file_size_mb:.2f} MB")
    print(f"  模型输入尺寸          : {cam_w}x{cam_h}")
    print(f"  模型加载耗时          : {load_time} ms")
    print(f"  实际测试帧数          : {n}")
    print(f"  总测试时间            : {total_ms / 1000:.2f} 秒")
    print("-" * 60)
    print(f"  平均 FPS              : {fps:.1f}")
    print("-" * 60)
    print(f"  每帧总耗时（取帧+推理）")
    print(f"    平均                : {avg_frame:.1f} ms")
    print(f"    中位数              : {median_f:.1f} ms")
    print(f"    最小                : {min(frame_times):.1f} ms")
    print(f"    最大                : {max(frame_times):.1f} ms")
    print(f"    P95                 : {p95_f:.1f} ms")
    print(f"    P99                 : {p99_f:.1f} ms")
    print("-" * 60)
    print(f"  推理耗时（前处理+TPU+后处理）")
    print(f"    平均                : {avg_infer:.1f} ms")
    print(f"    中位数              : {median_i:.1f} ms")
    print(f"    最小                : {min(infer_times):.1f} ms")
    print(f"    最大                : {max(infer_times):.1f} ms")
    print("-" * 60)
    print(f"  取帧耗时")
    print(f"    平均                : {avg_capture:.1f} ms")
    print("=" * 60)

    # 输出报告可直接复制的格式
    print()
    print()
    print(f"| 指标 | 值 |")
    print(f"|---|---|")
    print(f"| 模型文件 | {model_label} |")
    print(f"| 模型大小 | {file_size_mb:.2f} MB |")
    print(f"| 输入尺寸 | {cam_w}x{cam_h} |")
    print(f"| 平均 FPS | {fps:.1f} |")
    print(f"| 每帧平均耗时 | {avg_frame:.1f} ms |")
    print(f"| 每帧中位数耗时 | {median_f:.1f} ms |")
    print(f"| 推理平均耗时 | {avg_infer:.1f} ms |")
    print(f"| P95 帧耗时 | {p95_f:.1f} ms |")
    print(f"| P99 帧耗时 | {p99_f:.1f} ms |")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 benchmark.py <模型路径>")
        print("示例: python3 benchmark.py /path/to/yolo26n_seg_bf16.cvimodel")
        sys.exit(1)

    run_benchmark(sys.argv[1])
