#include <stdint.h>
#include <string.h>
#include <sys/time.h>

#include "yolov8.h"
#include "rknnPool.hpp"

#include <mutex>
#include <atomic>

// #define USE_RTSP // 是否使用rtsp推流

std::atomic<bool> running(true);

void capture_thread(cv::VideoCapture& capture, rknnPool<rkYolov8, cv::Mat&, All_result>& testPool)
{
    while (running && capture.isOpened())
    {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        // 丢进线程池
        if (testPool.put(frame) != 0)
        {
            printf("Put frame to pool failed!\n");
            break;
        }
    }
}

void display_thread(rknnPool<rkYolov8, cv::Mat&, All_result>& testPool, int threadNum)
{
#ifdef USE_RTSP

    int width = 1280;
    int height = 720;
    int fps = 60;

    // FFmpeg 推流命令
    std::string cmd =
        "ffmpeg -y "
        "-f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height) +
        " -r " + std::to_string(fps) +
        " -i - "
        "-c:v h264_rkmpp -preset ultrafast -tune zerolatency "
        "-fflags nobuffer -flags low_delay "
        "-rtsp_transport udp "
        "-f rtsp rtsp://10.60.90.188:8554/video";

    FILE* ffmpeg = popen(cmd.c_str(), "w");
    if (!ffmpeg) {
        std::cerr << "Failed to open ffmpeg pipe!" << std::endl;
        return;
    }

#endif 
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto beforeTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int count = 0;
    
    bool start_get = false;
    All_result result;
    while (running)
    {
        if (testPool.get(result) == 0 && !result.img.empty()) {
#ifdef USE_RTSP
            fwrite(result.img.data, 1, width * height * 3, ffmpeg);
#else

            // 显示帧
            cv::imshow("Yolov8", result.img);
            char c = (char)cv::waitKey(1);
            if (c == 'q' || c == 'Q') {
                break;
            }
#endif

            count++;
            if (count >= 60) {
                gettimeofday(&time, nullptr);
                auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
                printf("当前帧率: %.2f fps\n", 60.0 / float(currentTime - beforeTime) * 1000.0);
                beforeTime = currentTime;
                count = 0;
            }
        }
    }
#ifdef USE_RTSP
    pclose(ffmpeg);
#endif
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("%s <model_path> <video_path or camera_id>\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* video_name = argv[2];

    int threadNum = 3;
    rknnPool<rkYolov8, cv::Mat&, All_result> testPool(model_path, threadNum);

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
        // cap.open(std::string(video_name));
    }

    else
    {
        cap.open(std::string(video_name));
    }

    // 线程同步
    std::thread t1(capture_thread, std::ref(cap), std::ref(testPool));
    std::thread t2(display_thread, std::ref(testPool), threadNum);

    t1.join();
    t2.join();

    cap.release();
    cv::destroyAllWindows();
    deinit_post_process();

    return 0;
}