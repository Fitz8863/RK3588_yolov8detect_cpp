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

#include <mutex>
#include <atomic>


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <video_path or camera_id>\n", argv[0]);
        printf("Example: %s model.rknn 0   # use camera\n", argv[0]);
        printf("         %s model.rknn video.mp4   # use video file\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_name = argv[2];

    int threadNum = 3;
    rknnPool<rkYolov8, cv::Mat &, object_detect_result_list> testPool(model_path, threadNum);

    init_post_process();

    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
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
        if (frame.empty())
            break;
        object_detect_result_list od_results;

        if (testPool.put(frame) != 0)
        {
            printf("Put frame to pool failed!\n");
            break;
        }

        if (count >= threadNum && testPool.get(od_results) != 0)
            break;


        // 显示
        cv::imshow("YOLOv8 RKNN Detection", frame);

        // 按 q 退出
        if (cv::waitKey(1) == 'q')
            break;

        count++;
        if (count >= 60)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("60帧平均帧率: %.2f fps\n", 60.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
            count = 0;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    deinit_post_process();

    return 0;
}