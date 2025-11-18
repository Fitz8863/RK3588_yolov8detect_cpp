// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _RKNN_DEMO_YOLOV8_H_
#define _RKNN_DEMO_YOLOV8_H_

#include "rknn_api.h"
#include "common.h"
#include <string>
#include <mutex>
#include "postprocess.h"
#include "image_drawing.h"

class rkYolov8
{
private:
    std::mutex mtx;
    image_buffer_t dst_img;
    std::string model_path;
    // char* model_path;
    float nms_threshold, box_conf_threshold;     // 默认的NMS阈值   // 默认的置信度阈值
    rknn_app_context_t app_ctx;

public:

    rkYolov8(const char* model_path);
    int init_yolov8_model(rknn_app_context_t* input_app_ctx,bool share_weight);
    rknn_app_context_t *Get_app_ctx();
    All_result inference_yolov8_model(cv::Mat &frame);
    ~rkYolov8();
};



#endif //_RKNN_DEMO_YOLOV8_H_