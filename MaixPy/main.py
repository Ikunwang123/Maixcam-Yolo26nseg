from maix import app, camera, display, image, time
import numpy as np

from yolo26_seg_maixpy import YOLO26Seg


def get_blind_lane_hint(objs, img_w: int) -> str:
    if not objs:
        return "Blind lane: NOT FOUND"

    best = objs[0]
    mask = best.seg_mask
    if mask is None or mask.size == 0:
        return "Blind lane: NOT FOUND"

    h, w = mask.shape
    lower = mask[h // 2 :, :]
    if lower.size == 0:
        lower = mask

    valid = lower > 0
    total = int(valid.sum())
    if total < max(20, lower.size // 220):
        return "Blind lane: WEAK"

    third = max(1, w // 3)
    left = int(valid[:, :third].sum())
    mid = int(valid[:, third : min(2 * third, w)].sum())
    right = int(valid[:, min(2 * third, w) :].sum())

    col = valid.sum(axis=0).astype(np.float32)
    if float(col.sum()) > 1e-6:
        xs = np.arange(w, dtype=np.float32)
        local_center = float((xs * col).sum() / col.sum())
    else:
        local_center = float(w) * 0.5

    global_ratio = (best.x + local_center) / max(1.0, float(img_w - 1))

    if global_ratio < 0.33 or left > (mid + right):
        return "Blind lane: LEFT"
    if global_ratio > 0.67 or right > (left + mid):
        return "Blind lane: RIGHT"
    if left > right * 1.35:
        return "Blind lane: CENTER-LEFT"
    if right > left * 1.35:
        return "Blind lane: CENTER-RIGHT"
    return "Blind lane: CENTER"


def main() -> None:
    detector = YOLO26Seg("/root/.picoclaw/workspace/skills/blind-lane-assistant/yolo26n_seg_int8.cvimodel", default_input_w=640, default_input_h=640)
    detector.fast_single_class = True
    detector.max_candidates_before_nms = 80
    detector.max_results_after_nms = 1
    detector.enable_sparse_overlay = False

    try:
        cam_fmt = image.Format.FMT_RGB888
    except AttributeError:
        cam_fmt = image.FMT_RGB888

    cam = camera.Camera(detector.input_width(), detector.input_height(), cam_fmt)
    disp = display.Display()

    frame_cnt = 0
    frame_id = 0
    last_print_ms = time.ticks_ms()
    last_mask_info = "mask0_ratio:0.000"
    last_hint = "Blind lane: NOT FOUND"

    while not app.need_exit():
        img = cam.read()
        if img is None:
            continue
        frame_id += 1

        objs = detector.detect(img, conf_th=0.50, iou_th=0.50)

        if objs:
            obj = objs[0]
            detector.draw_seg_mask(img, obj.x, obj.y, obj.seg_mask, threshold=127)
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED, thickness=2)

        if frame_id % 2 == 0:
            last_hint = get_blind_lane_hint(objs, img.width())

        img.draw_rect(0, 0, img.width(), 26, color=image.COLOR_BLACK, thickness=-1)
        img.draw_string(4, 4, last_hint, image.COLOR_GREEN)

        if objs and frame_id % 2 == 0:
            mask0 = objs[0].seg_mask
            ratio = float((mask0 > 0).sum()) / float(max(1, mask0.size))
            last_mask_info = "mask0_ratio:{:.3f}".format(ratio)
        elif not objs:
            last_mask_info = "mask0_ratio:0.000"

        disp.show(img)

        frame_cnt += 1
        now_ms = time.ticks_ms()
        if now_ms - last_print_ms >= 1000:
            elapsed = max(1, now_ms - last_print_ms)
            fps = frame_cnt * 1000.0 / elapsed
            print("FPS: {:.2f} | {} | {}".format(fps, last_mask_info, last_hint))
            frame_cnt = 0
            last_print_ms = now_ms


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print(traceback.format_exc())
        while not app.need_exit():
            time.sleep_ms(100)



