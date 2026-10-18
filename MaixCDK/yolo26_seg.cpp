#include "yolo26_seg.hpp"
#include <omp.h>

// ====================== utils ======================
static float calc_iou(const cv::Rect& a, const cv::Rect& b) {
    int x_left = std::max(a.x, b.x);
    int y_top = std::max(a.y, b.y);
    int x_right = std::min(a.x + a.width,  b.x + b.width);
    int y_bottom = std::min(a.y + a.height, b.y + b.height);

    if (x_right <= x_left || y_bottom <= y_top) return 0.0f;

    float inter = float(x_right - x_left) * float(y_bottom - y_top);
    float area_a = float(std::max(0, a.width)) * float(std::max(0, a.height));
    float area_b = float(std::max(0, b.width)) * float(std::max(0, b.height));
    float uni = area_a + area_b - inter;
    if (uni <= 1e-6f) return 0.0f;
    return inter / uni;
}

static void apply_nms(std::vector<YOLO26Seg::Candidate>& candidates,
                      float iou_th,
                      std::vector<YOLO26Seg::Candidate>& results) {
    std::sort(candidates.begin(), candidates.end(),
              [](const YOLO26Seg::Candidate& a, const YOLO26Seg::Candidate& b) {
                  return a.score > b.score;
              });

    std::vector<uint8_t> suppressed(candidates.size(), 0);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            // 若你要按类做NMS，可加: if (candidates[i].class_id != candidates[j].class_id) continue;
            if (calc_iou(candidates[i].box, candidates[j].box) > iou_th) {
                suppressed[j] = 1;
            }
        }
    }
}

// ====================== class ======================
YOLO26Seg::YOLO26Seg(const std::string& model_path, int default_input_w, int default_input_h)
    : model(nullptr), input_w(default_input_w), input_h(default_input_h) {
    model = new maix::nn::NN(model_path, true);

    // 尝试从模型输入shape读取真实输入尺寸
    // 这里通过 forward 前拿不到输入列表时，保持默认值；实际中你可按 maix 的API补充更稳妥获取方式
    // 目前先使用构造参数 + detect时按输出头反推兼容
}

YOLO26Seg::~YOLO26Seg() {
    if (model) {
        delete model;
        model = nullptr;
    }
}

void YOLO26Seg::decode_head(maix::tensor::Tensor* cls_t,
                            maix::tensor::Tensor* box_t,
                            maix::tensor::Tensor* mask_t,
                            int stride_x,
                            int stride_y,
                            float conf_th,
                            std::vector<Candidate>& candidates) {
    auto cls_shape = cls_t->shape();   // [1, Cc, H, W]
    auto box_shape = box_t->shape();   // [1, 4,  H, W]
    auto msk_shape = mask_t->shape();  // [1, 32, H, W]
    if (cls_shape.size() != 4 || box_shape.size() != 4 || msk_shape.size() != 4) return;

    int Cc = cls_shape[1];
    int H  = cls_shape[2];
    int W  = cls_shape[3];
    int hw = H * W;

    if (box_shape[2] != H || box_shape[3] != W) return;
    if (msk_shape[2] != H || msk_shape[3] != W) return;
    if (box_shape[1] != 4 || msk_shape[1] != 32) return;

    float* cls_data  = (float*)cls_t->data();
    float* box_data  = (float*)box_t->data();
    float* mask_data = (float*)mask_t->data();

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;

            float score = 0.0f;
            int class_id = 0;

            // 兼容单类(c=1)和多类(c>1)
            if (Cc == 1) {
                score = sigmoid(cls_data[idx]);
                class_id = 0;
            } else {
                float max_logit = cls_data[idx];
                int max_c = 0;
                for (int c = 1; c < Cc; ++c) {
                    float v = cls_data[c * hw + idx];
                    if (v > max_logit) {
                        max_logit = v;
                        max_c = c;
                    }
                }
                score = sigmoid(max_logit);
                class_id = max_c;
            }

            if (score < conf_th) continue;

            float l = box_data[0 * hw + idx];
            float t = box_data[1 * hw + idx];
            float r = box_data[2 * hw + idx];
            float b = box_data[3 * hw + idx];

            float cx = (x + 0.5f) * stride_x;
            float cy = (y + 0.5f) * stride_y;

            Candidate cand;
            cand.box.x      = (int)(cx - l * stride_x);
            cand.box.y      = (int)(cy - t * stride_y);
            cand.box.width  = (int)((l + r) * stride_x);
            cand.box.height = (int)((t + b) * stride_y);
            cand.score      = score;
            cand.class_id   = class_id;

            cand.mask_coefs.resize(32);
            for (int c = 0; c < 32; ++c) {
                cand.mask_coefs[c] = mask_data[c * hw + idx];
            }

            candidates.push_back(std::move(cand));
        }
    }
}

maix::nn::Objects* YOLO26Seg::detect(maix::image::Image& img, float conf_th, float iou_th) {
    int img_w = img.width();
    int img_h = img.height();

    // 先前向
    maix::tensor::Tensors* outputs = model->forward_image(
        img,
        {0.0f, 0.0f, 0.0f},
        {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f},
        maix::image::Fit::FIT_CONTAIN,
        true, false, true
    );

    maix::nn::Objects* objs = new maix::nn::Objects();
    if (!outputs) return objs;

    // 按输出shape分组
    std::map<std::pair<int,int>, maix::tensor::Tensor*> cls_map;   // key=(H,W)
    std::map<std::pair<int,int>, maix::tensor::Tensor*> box_map;
    std::map<std::pair<int,int>, maix::tensor::Tensor*> coef_map;
    maix::tensor::Tensor* proto = nullptr;

    // 先收集所有 c==32 的层，后面再分 proto / coef
    std::vector<maix::tensor::Tensor*> c32_list;

    for (auto& kv : *outputs) {
        maix::tensor::Tensor* t = kv.second;
        auto s = t->shape(); // [1, C, H, W]
        if (s.size() != 4) continue;
        int C = s[1], H = s[2], W = s[3];

        if (C == 4) {
            box_map[{H, W}] = t;
        } else if (C == 1 || C > 4) {
            // 你当前模型是 C=1；多类时也可走这里
            if (C != 32) cls_map[{H, W}] = t;
            else c32_list.push_back(t);
        }
    }

    // 从 c32 里分 proto 和 mask coef:
    // 与 box 同分辨率的 c32 归为 coef；剩下最大的那个归 proto
    int best_area = -1;
    for (auto* t : c32_list) {
        auto s = t->shape();
        int H = s[2], W = s[3];
        auto key = std::make_pair(H, W);
        if (box_map.count(key)) {
            coef_map[key] = t;
        } else {
            int area = H * W;
            if (area > best_area) {
                best_area = area;
                proto = t;
            }
        }
    }

    // 如果 proto 还没找到，兜底：选 c32 最大分辨率
    if (!proto && !c32_list.empty()) {
        int max_area = -1;
        for (auto* t : c32_list) {
            auto s = t->shape();
            int area = s[2] * s[3];
            if (area > max_area) {
                max_area = area;
                proto = t;
            }
        }
    }

    // 从检测头反推输入尺寸（更兼容 320x224 / 640x640）
    // 取最大检测层，通常stride=8 => input ~= H*8, W*8
    int max_h = -1, max_w = -1;
    for (auto& kv : box_map) {
        if (kv.first.first * kv.first.second > max_h * max_w) {
            max_h = kv.first.first;
            max_w = kv.first.second;
        }
    }
    if (max_h > 0 && max_w > 0) {
        input_h = max_h * 8;
        input_w = max_w * 8;
    }

    // letterbox参数（按真实 input_w/input_h）
    float scale = std::min((float)input_w / (float)img_w, (float)input_h / (float)img_h);
    float pad_w = ((float)input_w - (float)img_w * scale) * 0.5f;
    float pad_h = ((float)input_h - (float)img_h * scale) * 0.5f;

    // 解码所有可配对检测层
    std::vector<Candidate> all_candidates;
    for (auto& kv : box_map) {
        auto key = kv.first;
        int H = key.first, W = key.second;

        if (!cls_map.count(key)) continue;
        if (!coef_map.count(key)) continue;

        int stride_y = std::max(1, input_h / H);
        int stride_x = std::max(1, input_w / W);

        decode_head(cls_map[key], box_map[key], coef_map[key],
                    stride_x, stride_y, conf_th, all_candidates);
    }

    if (all_candidates.empty() || !proto) {
        delete outputs;
        return objs;
    }

    // NMS
    std::vector<Candidate> final_results;
    apply_nms(all_candidates, iou_th, final_results);

    // proto shape
    auto ps = proto->shape(); // [1,32,mask_h,mask_w]
    if (ps.size() != 4 || ps[1] != 32) {
        delete outputs;
        return objs;
    }
    int mask_h = ps[2];
    int mask_w = ps[3];
    int mask_num = 32;
    int mask_square = mask_h * mask_w;
    float* proto_data = (float*)proto->data();

    for (const auto& cand : final_results) {
        // 模型输入空间 -> 原图空间
        int orig_x = (int)((cand.box.x - pad_w) / scale);
        int orig_y = (int)((cand.box.y - pad_h) / scale);
        int orig_w = (int)(cand.box.width / scale);
        int orig_h = (int)(cand.box.height / scale);

        // 裁剪
        orig_x = std::max(0, std::min(orig_x, img_w - 1));
        orig_y = std::max(0, std::min(orig_y, img_h - 1));
        orig_w = std::min(orig_w, img_w - orig_x);
        orig_h = std::min(orig_h, img_h - orig_y);
        if (orig_w <= 0 || orig_h <= 0) continue;

        // 在 proto 空间截取 bbox 区域
        int x1 = std::max(0, cand.box.x);
        int y1 = std::max(0, cand.box.y);
        int x2 = std::min(input_w, cand.box.x + cand.box.width);
        int y2 = std::min(input_h, cand.box.y + cand.box.height);
        if (x2 <= x1 || y2 <= y1) continue;

        int m_x1 = std::max(0, x1 * mask_w / input_w);
        int m_y1 = std::max(0, y1 * mask_h / input_h);
        int m_x2 = std::min(mask_w, x2 * mask_w / input_w);
        int m_y2 = std::min(mask_h, y2 * mask_h / input_h);

        int crop_mw = m_x2 - m_x1;
        int crop_mh = m_y2 - m_y1;
        if (crop_mw <= 0 || crop_mh <= 0) continue;

        cv::Mat small_mask(crop_mh, crop_mw, CV_8UC1);

        for (int r = 0; r < crop_mh; ++r) {
            int gr = m_y1 + r;
            for (int c = 0; c < crop_mw; ++c) {
                int gc = m_x1 + c;
                float sum = 0.0f;
                int pix = gr * mask_w + gc;
                for (int n = 0; n < mask_num; ++n) {
                    sum += cand.mask_coefs[n] * proto_data[n * mask_square + pix];
                }
                small_mask.at<uint8_t>(r, c) = (sum > 0.0f) ? 255 : 0;
            }
        }

        cv::Mat resized_mask;
        cv::resize(small_mask, resized_mask, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

        maix::image::Image* seg_img = new maix::image::Image(
            orig_w, orig_h, maix::image::FMT_GRAYSCALE,
            resized_mask.data, orig_w * orig_h, true
        );

        maix::nn::Object& obj = objs->add(orig_x, orig_y, orig_w, orig_h, cand.class_id, cand.score);
        obj.seg_mask = seg_img;
    }

    delete outputs;
    return objs;
}

void YOLO26Seg::draw_seg_mask(maix::image::Image& img, int x, int y, maix::image::Image* mask, int threshold) {
    if (!mask || mask->format() != maix::image::FMT_GRAYSCALE) return;

    int w = std::min(mask->width(),  img.width()  - x);
    int h = std::min(mask->height(), img.height() - y);
    if (w <= 0 || h <= 0) return;

    uint8_t* img_ptr  = (uint8_t*)img.data();
    uint8_t* mask_ptr = (uint8_t*)mask->data();

    int img_w  = img.width();
    int mask_w = mask->width();

    bool is_rgb = (img.format() == maix::image::FMT_RGB888);
    int c_idx = is_rgb ? 0 : 2; // RGB:0(BGR里红色通道是2)

    #pragma omp parallel for
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            if (mask_ptr[r * mask_w + c] > threshold) {
                int idx = ((y + r) * img_w + (x + c)) * 3;
                img_ptr[idx + c_idx] = (img_ptr[idx + c_idx] >> 1) + 127;
            }
        }
    }
}
