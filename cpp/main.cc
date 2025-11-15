#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <sys/time.h>

#include "yolov8.h"
#include "rknnPool.hpp"

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
    const char* video_name  = argv[2];

    int threadNum = 3;
    // rknnPool<rkYolov8, image_buffer_t*, object_detect_result_list> testPool(model_path, threadNum);

    // std::cout<<"比亚迪的"<<std::endl;

    int ret;
    // rknn_app_context_t rknn_app_ctx;
    // memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    rkYolov8 yolo(model_path);

    // if (testPool.init() != 0)
    // {
    //     printf("rknnPool init fail!\n");
    //     return -1;
    // }

    ret = yolo.init_yolov8_model(yolo.Get_app_ctx(),false);
    
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    // 打开视频或摄像头
    cv::VideoCapture cap;
    if (std::string(video_name).find("/dev/video") == 0)
    {
        
        std::string pipeline = "v4l2src device=" + std::string(video_name) +
            " ! image/jpeg, width=1280, height=720, framerate=60/1 ! "
            "jpegdec ! videoconvert ! appsink";
        cap.open(pipeline, cv::CAP_GSTREAMER);

        // 如果没有GStreamer环境的话使用下面这个
        // capture.open(std::string(video_name));
 
    }
    else
    {
        cap.open(std::string(video_name));
    }

    struct timeval time;
    auto beforeTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int count = 0;

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

        // ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);

        od_results = yolo.inference_yolov8_model(&src_image);

        // if (testPool.put(&src_image) != 0)
        // {
        //     printf("Put frame to pool failed!\n");
        //     break;
        // }

        // if(testPool.get(od_results) == 0){

        // }

        if (ret != 0)
        {
            printf("inference_yolov8_model fail! ret=%d\n", ret);
            break;
        }

    
        // 显示
        cv::imshow("YOLOv8 RKNN Detection", frame);

        // 按 q 退出
        if (cv::waitKey(1) == 'q')
            break;

        count++;
        if (count >= 60) {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("60帧平均帧率: %.2f fps\n", 60.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
            count = 0;
        }

    }

    deinit_post_process();

    return 0;
}