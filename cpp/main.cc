#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <video_path or camera_id>\n", argv[0]);
        printf("Example: %s model.rknn 0   # use camera\n", argv[0]);
        printf("         %s model.rknn video.mp4   # use video file\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    std::string input_source = argv[2];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    // 初始化模型
    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    // 打开视频或摄像头
    cv::VideoCapture cap;
    if (isdigit(input_source[0]))
    {
        int cam_id = std::stoi(input_source);
        cap.open(cam_id);
    }
    else
    {
        cap.open(input_source);
    }

    if (!cap.isOpened())
    {
        printf("Failed to open video/camera: %s\n", input_source.c_str());
        return -1;
    }

    struct timeval time;
    auto beforeTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // 将cv::Mat转换为image_buffer_t
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));

        src_image.width = frame.cols;
        src_image.height = frame.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        
        src_image.size = frame.cols * frame.rows * 3;
        src_image.virt_addr = frame.data; // 不复制数据，直接引用OpenCV内存

        object_detect_result_list od_results;
        memset(&od_results, 0, sizeof(od_results));

        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);

        if (ret != 0)
        {
            printf("inference_yolov8_model fail! ret=%d\n", ret);
            break;
        }

        // 绘制检测框
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result* det_result = &(od_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 255, 0), 2);
            char text[128];
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            cv::putText(frame, text, cv::Point(x1, y1 - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            // std::cout << text;
        }
        // std::cout << "\n" << std::endl;

    
        // 显示
        cv::imshow("YOLOv8 RKNN Detection", frame);

        // 按 q 退出
        if (cv::waitKey(1) == 'q')
            break;

        frames++;
        if (frames >= 60) {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("60帧平均帧率: %.2f fps\n", 60.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
            frames = 0;
        }

    }

    deinit_post_process();
    release_yolov8_model(&rknn_app_ctx);

    return 0;
}