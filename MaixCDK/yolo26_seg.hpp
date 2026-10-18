#pragma once

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include "maix_nn.hpp"
#include "maix_image.hpp"
#include "maix_nn_object.hpp"
#include <opencv2/opencv.hpp>

class YOLO26Seg {
public:
    struct Candidate {
        cv::Rect box;                  // in model input space
        float score;
        int class_id;
        std::vector<float> mask_coefs; // 32-dim
    };

    /**
     * @brief 初始化 YOLO26Seg 分割模型
     * @param model_path .cvimodel 模型文件路径
     * @param default_input_w 默认输入宽（仅在无法从模型读取时使用）
     * @param default_input_h 默认输入高（仅在无法从模型读取时使用）
     */
    YOLO26Seg(const std::string& model_path, int default_input_w = 640, int default_input_h = 640);
    ~YOLO26Seg();

    /**
     * @brief 执行目标检测与分割
     */
    maix::nn::Objects* detect(maix::image::Image& img, float conf_th = 0.45f, float iou_th = 0.45f);

    /**
     * @brief 将分割 Mask 画到图像上
     */
    void draw_seg_mask(maix::image::Image& img, int x, int y, maix::image::Image* mask, int threshold = 127);

private:
    maix::nn::NN* model;
    int input_w;
    int input_h;

    inline float sigmoid(float x) {
        x = std::max(-88.0f, std::min(88.0f, x));
        return 1.0f / (1.0f + std::exp(-x));
    }

    // 解码单个检测头
    void decode_head(maix::tensor::Tensor* cls_t,
                     maix::tensor::Tensor* box_t,
                     maix::tensor::Tensor* mask_t,
                     int stride_x,
                     int stride_y,
                     float conf_th,
                     std::vector<Candidate>& candidates);
};
