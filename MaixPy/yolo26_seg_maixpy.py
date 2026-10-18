from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from maix import image, nn, tensor


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    class_id: int
    mask_coefs: np.ndarray


@dataclass
class SegObject:
    x: int
    y: int
    w: int
    h: int
    class_id: int
    score: float
    seg_mask: np.ndarray


class YOLO26Seg:
    def __init__(self, model_path: str, default_input_w: int = 640, default_input_h: int = 640, dual_buff: bool = False):
        self.model = nn.NN(model_path, dual_buff=dual_buff)
        self.input_w = int(default_input_w)
        self.input_h = int(default_input_h)
        self.mean = [0.0, 0.0, 0.0]
        self.scale = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        self.max_candidates_before_nms = 180
        self.max_results_after_nms = 6
        self.enable_sparse_overlay = False
        self.sparse_overlay_points = 90
        self.fast_single_class = True
        self.print_debug_once = True
        self._debug_printed = False
        self._try_read_input_size()

    def input_width(self) -> int:
        return self.input_w

    def input_height(self) -> int:
        return self.input_h

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-88.0, min(88.0, float(x)))
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _calc_iou(a: Candidate, b: Candidate) -> float:
        ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
        bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0

        area_a = float(max(0, a.w) * max(0, a.h))
        area_b = float(max(0, b.w) * max(0, b.h))
        uni = area_a + area_b - inter
        if uni <= 1e-6:
            return 0.0
        return inter / uni

    def _try_read_input_size(self) -> None:
        try:
            infos = self.model.inputs_info()
            if not infos:
                return
            shape = infos[0].shape
            if len(shape) != 4:
                return

            # Commonly NCHW: [N, C, H, W].
            if int(shape[1]) <= 8:
                self.input_h = int(shape[2])
                self.input_w = int(shape[3])
            else:
                # Fallback for NHWC-like shape.
                self.input_h = int(shape[1])
                self.input_w = int(shape[2])
        except Exception:
            pass

    @staticmethod
    def _to_nchw(arr: np.ndarray) -> Optional[np.ndarray]:
        if arr is None or arr.ndim != 4:
            return None

        # Already NCHW in most MaixPy NN outputs.
        if arr.shape[1] <= 64:
            return arr

        # Fallback: NHWC -> NCHW.
        if arr.shape[3] <= 64:
            return np.transpose(arr, (0, 3, 1, 2))

        return arr

    def _forward_image(self, img: image.Image):
        try:
            return self.model.forward_image(
                img,
                self.mean,
                self.scale,
                fit=image.Fit.FIT_CONTAIN,
                copy_result=False,
            )
        except TypeError:
            pass

        try:
            return self.model.forward_image(img, self.mean, self.scale, copy_result=False)
        except TypeError:
            pass

        try:
            return self.model.forward_image(img, self.mean, self.scale)
        except TypeError:
            pass

        return self.model.forward_image(img)

    @staticmethod
    def _apply_nms(candidates: List[Candidate], iou_th: float) -> List[Candidate]:
        if not candidates:
            return []

        cands = sorted(candidates, key=lambda x: x.score, reverse=True)
        keep: List[Candidate] = []
        suppressed = [False] * len(cands)

        for i, ci in enumerate(cands):
            if suppressed[i]:
                continue
            keep.append(ci)
            for j in range(i + 1, len(cands)):
                if suppressed[j]:
                    continue
                if YOLO26Seg._calc_iou(ci, cands[j]) > iou_th:
                    suppressed[j] = True

        return keep

    def _decode_head(
        self,
        cls_nchw: np.ndarray,
        box_nchw: np.ndarray,
        coef_nchw: np.ndarray,
        stride_x: int,
        stride_y: int,
        conf_th: float,
        out: List[Candidate],
    ) -> None:
        cls_map = cls_nchw[0]
        box_map = box_nchw[0]
        coef_map = coef_nchw[0]

        cc, h, w = cls_map.shape
        hw = h * w

        if box_map.shape != (4, h, w):
            return
        if coef_map.shape != (32, h, w):
            return

        cls_flat = cls_map.reshape(cc, hw)
        box_flat = box_map.reshape(4, hw)
        coef_flat = coef_map.reshape(32, hw)

        for idx in range(hw):
            if cc == 1:
                score = float(self._sigmoid(cls_flat[0, idx]))
                class_id = 0
            else:
                logits = cls_flat[:, idx]
                class_id = int(np.argmax(logits))
                score = float(self._sigmoid(logits[class_id]))

            if score < conf_th:
                continue

            gy = idx // w
            gx = idx % w

            l = float(box_flat[0, idx])
            t = float(box_flat[1, idx])
            r = float(box_flat[2, idx])
            b = float(box_flat[3, idx])

            cx = (gx + 0.5) * stride_x
            cy = (gy + 0.5) * stride_y

            x = int(cx - l * stride_x)
            y = int(cy - t * stride_y)
            bw = int((l + r) * stride_x)
            bh = int((t + b) * stride_y)

            out.append(
                Candidate(
                    x=x,
                    y=y,
                    w=bw,
                    h=bh,
                    score=score,
                    class_id=class_id,
                    mask_coefs=coef_flat[:, idx].copy(),
                )
            )

    def _decode_head_fast_single(
        self,
        cls_nchw: np.ndarray,
        box_nchw: np.ndarray,
        coef_nchw: np.ndarray,
        stride_x: int,
        stride_y: int,
        conf_th: float,
        out: List[Candidate],
    ) -> None:
        cls_map = cls_nchw[0]
        box_map = box_nchw[0]
        coef_map = coef_nchw[0]

        cc, h, w = cls_map.shape
        if cc != 1:
            return
        if box_map.shape != (4, h, w):
            return
        if coef_map.shape != (32, h, w):
            return

        score_map = 1.0 / (1.0 + np.exp(-cls_map[0]))
        flat_scores = score_map.reshape(-1)
        best_idx = int(np.argmax(flat_scores))
        best_score = float(flat_scores[best_idx])
        if best_score < conf_th:
            return

        gy = best_idx // w
        gx = best_idx % w

        l = float(box_map[0, gy, gx])
        t = float(box_map[1, gy, gx])
        r = float(box_map[2, gy, gx])
        b = float(box_map[3, gy, gx])

        cx = (gx + 0.5) * stride_x
        cy = (gy + 0.5) * stride_y

        x = int(cx - l * stride_x)
        y = int(cy - t * stride_y)
        bw = int((l + r) * stride_x)
        bh = int((t + b) * stride_y)

        out.append(
            Candidate(
                x=x,
                y=y,
                w=bw,
                h=bh,
                score=best_score,
                class_id=0,
                mask_coefs=coef_map[:, gy, gx].copy(),
            )
        )

    @staticmethod
    def _resize_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        in_h, in_w = mask.shape
        if in_h == out_h and in_w == out_w:
            return mask

        ys = (np.arange(out_h) * (in_h / max(1, out_h))).astype(np.int32)
        xs = (np.arange(out_w) * (in_w / max(1, out_w))).astype(np.int32)
        ys = np.clip(ys, 0, in_h - 1)
        xs = np.clip(xs, 0, in_w - 1)
        return mask[ys[:, None], xs[None, :]]

    def detect(self, img: image.Image, conf_th: float = 0.45, iou_th: float = 0.45) -> List[SegObject]:
        img_w = int(img.width())
        img_h = int(img.height())

        outputs = self._forward_image(img)
        if outputs is None:
            return []

        if hasattr(outputs, "items"):
            output_iter = outputs.items()
        elif hasattr(outputs, "keys"):
            output_iter = ((k, outputs[k]) for k in outputs.keys())
        else:
            return []

        cls_map: Dict[Tuple[int, int], np.ndarray] = {}
        box_map: Dict[Tuple[int, int], np.ndarray] = {}
        coef_map: Dict[Tuple[int, int], np.ndarray] = {}
        c32_list: List[np.ndarray] = []
        proto: Optional[np.ndarray] = None
        output_shapes: List[str] = []

        for _, t in output_iter:
            arr = tensor.tensor_to_numpy_float32(t, copy=False)
            arr_nchw = self._to_nchw(arr)
            if arr_nchw is None:
                continue

            c = int(arr_nchw.shape[1])
            h = int(arr_nchw.shape[2])
            w = int(arr_nchw.shape[3])
            output_shapes.append("[1,{},{},{}]".format(c, h, w))
            key = (h, w)

            if c == 4:
                box_map[key] = arr_nchw
            elif c == 32:
                c32_list.append(arr_nchw)
            else:
                cls_map[key] = arr_nchw

        best_area = -1
        for t in c32_list:
            h, w = int(t.shape[2]), int(t.shape[3])
            key = (h, w)
            if key in box_map:
                coef_map[key] = t
            else:
                area = h * w
                if area > best_area:
                    best_area = area
                    proto = t

        if proto is None and c32_list:
            proto = max(c32_list, key=lambda x: int(x.shape[2]) * int(x.shape[3]))

        if self.print_debug_once and not self._debug_printed:
            print("[YOLO26Seg] outputs:", ", ".join(output_shapes))
            print("[YOLO26Seg] box heads:", sorted(box_map.keys()))
            print("[YOLO26Seg] cls heads:", sorted(cls_map.keys()))
            print("[YOLO26Seg] coef heads:", sorted(coef_map.keys()))
            if proto is not None:
                print("[YOLO26Seg] proto shape:", proto.shape)
            else:
                print("[YOLO26Seg] proto shape: None")
            self._debug_printed = True

        if box_map:
            max_key = max(box_map.keys(), key=lambda k: k[0] * k[1])
            self.input_h = int(max_key[0] * 8)
            self.input_w = int(max_key[1] * 8)

        scale = min(float(self.input_w) / float(img_w), float(self.input_h) / float(img_h))
        pad_w = (float(self.input_w) - float(img_w) * scale) * 0.5
        pad_h = (float(self.input_h) - float(img_h) * scale) * 0.5

        all_candidates: List[Candidate] = []
        for key, box_t in box_map.items():
            if key not in cls_map or key not in coef_map:
                continue

            h, w = key
            stride_y = max(1, self.input_h // h)
            stride_x = max(1, self.input_w // w)
            cls_channels = int(cls_map[key].shape[1])
            if self.fast_single_class and cls_channels == 1:
                self._decode_head_fast_single(cls_map[key], box_t, coef_map[key], stride_x, stride_y, conf_th, all_candidates)
            else:
                self._decode_head(cls_map[key], box_t, coef_map[key], stride_x, stride_y, conf_th, all_candidates)

        if not all_candidates or proto is None:
            return []

        if len(all_candidates) > self.max_candidates_before_nms:
            all_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)[: self.max_candidates_before_nms]

        if self.fast_single_class:
            final_candidates = [max(all_candidates, key=lambda x: x.score)]
        else:
            final_candidates = self._apply_nms(all_candidates, iou_th)
        if not final_candidates:
            return []
        if len(final_candidates) > self.max_results_after_nms:
            final_candidates = final_candidates[: self.max_results_after_nms]

        proto_map = proto[0]
        if proto_map.shape[0] != 32:
            return []

        mask_h = int(proto_map.shape[1])
        mask_w = int(proto_map.shape[2])

        results: List[SegObject] = []
        for cand in final_candidates:
            orig_x = int((cand.x - pad_w) / scale)
            orig_y = int((cand.y - pad_h) / scale)
            orig_w = int(cand.w / scale)
            orig_h = int(cand.h / scale)

            orig_x = max(0, min(orig_x, img_w - 1))
            orig_y = max(0, min(orig_y, img_h - 1))
            orig_w = min(orig_w, img_w - orig_x)
            orig_h = min(orig_h, img_h - orig_y)
            if orig_w <= 0 or orig_h <= 0:
                continue

            x1 = max(0, cand.x)
            y1 = max(0, cand.y)
            x2 = min(self.input_w, cand.x + cand.w)
            y2 = min(self.input_h, cand.y + cand.h)
            if x2 <= x1 or y2 <= y1:
                continue

            mx1 = max(0, x1 * mask_w // self.input_w)
            my1 = max(0, y1 * mask_h // self.input_h)
            mx2 = min(mask_w, x2 * mask_w // self.input_w)
            my2 = min(mask_h, y2 * mask_h // self.input_h)

            crop_w = mx2 - mx1
            crop_h = my2 - my1
            if crop_w <= 0 or crop_h <= 0:
                continue

            proto_crop = proto_map[:, my1:my2, mx1:mx2]
            coef = cand.mask_coefs.reshape(32, 1, 1)
            small_logits = np.sum(proto_crop * coef, axis=0)
            small_prob = 1.0 / (1.0 + np.exp(-small_logits))
            small_mask = np.where(small_prob > 0.50, 255, 0).astype(np.uint8)

            # Relax threshold progressively to avoid empty mask on some int8 exports.
            if np.count_nonzero(small_mask) == 0:
                small_mask = np.where(small_prob > 0.35, 255, 0).astype(np.uint8)
            if np.count_nonzero(small_mask) == 0:
                dyn_th = float(np.max(small_prob) * 0.70)
                small_mask = np.where(small_prob >= dyn_th, 255, 0).astype(np.uint8)
            if np.count_nonzero(small_mask) == 0:
                # Last fallback: keep region visible for debugging instead of showing only box.
                small_mask = np.full((crop_h, crop_w), 255, dtype=np.uint8)

            resized_mask = self._resize_nearest(small_mask, orig_w, orig_h)

            results.append(
                SegObject(
                    x=orig_x,
                    y=orig_y,
                    w=orig_w,
                    h=orig_h,
                    class_id=int(cand.class_id),
                    score=float(cand.score),
                    seg_mask=resized_mask,
                )
            )

        return results

    def draw_seg_mask(self, img: image.Image, x: int, y: int, mask: np.ndarray, threshold: int = 127) -> None:
        if mask is None or mask.ndim != 2:
            return

        cv_img = image.image2cv(img, copy=False)
        if cv_img is None or cv_img.ndim != 3 or cv_img.shape[2] < 3:
            return

        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        if x >= img_w or y >= img_h:
            return

        draw_w = min(mask.shape[1], img_w - x)
        draw_h = min(mask.shape[0], img_h - y)
        if draw_w <= 0 or draw_h <= 0:
            return

        roi = cv_img[y:y + draw_h, x:x + draw_w]
        mask_roi = mask[:draw_h, :draw_w] > threshold

        # Keep consistent with C++ behavior:
        # RGB image -> channel 0, BGR image -> channel 2.
        color_idx = 2
        try:
            fmt = img.format()
            if hasattr(image, "Format"):
                if fmt == image.Format.FMT_RGB888:
                    color_idx = 0
                elif fmt == image.Format.FMT_BGR888:
                    color_idx = 2
            else:
                if fmt == image.FMT_RGB888:
                    color_idx = 0
                elif fmt == image.FMT_BGR888:
                    color_idx = 2
        except Exception:
            color_idx = 2

        main_ch = roi[:, :, color_idx]
        main_ch[mask_roi] = ((main_ch[mask_roi] >> 1) + 127).astype(np.uint8)

        # Dim other channels so mask region is easier to see.
        for ch in range(3):
            if ch == color_idx:
                continue
            other = roi[:, :, ch]
            other[mask_roi] = (other[mask_roi] >> 1).astype(np.uint8)

        if self.enable_sparse_overlay:
            # Optional fallback for firmware where image2cv view update is weak.
            pts = np.argwhere(mask_roi)
            if pts.shape[0] > 0:
                draw_points = max(1, self.sparse_overlay_points)
                step = max(1, pts.shape[0] // draw_points)
                for i in range(0, pts.shape[0], step):
                    ry = int(pts[i, 0])
                    rx = int(pts[i, 1])
                    img.draw_rect(x + rx, y + ry, 2, 2, color=image.COLOR_RED, thickness=-1)
