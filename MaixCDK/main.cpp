#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_app.hpp"
#include "yolo26_seg.hpp"
#include "maix_basic.hpp"
#include "maix_time.hpp"
using namespace maix;

int main(int argc, char** argv) {
    YOLO26Seg detector("/root/.picoclaw/workspace/skills/blind-lane-assistant/yolo26n_seg_int8.cvimodel");

    camera::Camera cam(640, 640, image::FMT_RGB888);
    display::Display disp;

    // 开始帧率统计
    maix::time::fps_start();

    while (!app::need_exit()) {
        image::Image* img = cam.read();
        if (!img) continue;

        nn::Objects* objs = detector.detect(*img, 0.45f, 0.45f);

        if (objs) {
            for (auto obj : *objs) {
                detector.draw_seg_mask(*img, obj->x, obj->y, obj->seg_mask, 127);
                img->draw_rect(obj->x, obj->y, obj->w, obj->h, image::COLOR_RED, 2);
                std::string msg = "Class " + std::to_string(obj->class_id) + ": " + std::to_string(obj->score).substr(0,4);
                img->draw_string(obj->x, std::max(0, obj->y - 20), msg, image::COLOR_RED);
            }
        }

        disp.show(*img);

        // 每帧调用 fps()，更新内部时间记录
        float current_fps = maix::time::fps();

        // 每秒打印一次帧率
        static uint64_t last_print_ms = 0;
        uint64_t now_ms = maix::time::ticks_ms(); // 使用系统运行时间
        if (now_ms - last_print_ms >= 1000) {
            printf("FPS: %.2f\n", current_fps);
            last_print_ms = now_ms;
        }

        // 释放资源
        if (objs) delete objs;
        delete img;
    }
    return 0;
}

