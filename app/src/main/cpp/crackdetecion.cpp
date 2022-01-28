// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "crackdetection.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"


int CrackFinder::load( bool use_gpu)
{
    crackFinder.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    crackFinder.opt = ncnn::Option();

#if NCNN_VULKAN
    scrfd.opt.use_vulkan_compute = use_gpu;
#endif

    crackFinder.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "segmentation_model_192x192_ONNX_simpifyed-opt2.param");
    sprintf(modelpath, "segmentation_model_192x192_ONNX_simpifyed-opt2.bin");

    crackFinder.load_param(parampath);
    crackFinder.load_model(modelpath);


    return 0;
}

int CrackFinder::load(AAssetManager* mgr, bool use_gpu)
{
    scrfd.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    crackFinder.opt = ncnn::Option();

#if NCNN_VULKAN
    scrfd.opt.use_vulkan_compute = use_gpu;
#endif

    crackFinder.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "segmentation_model_192x192_ONNX_simpifyed-opt2.param");
    sprintf(modelpath, "segmentation_model_192x192_ONNX_simpifyed-opt2.bin");

    crackFinder.load_param(mgr, parampath);
    crackFinder.load_model(mgr, modelpath);

    has_kps = strstr(modeltype, "_kps") != NULL;

    return 0;
}

int CrackFinder::detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold)
{
    int target_width =224; int target_height =224;
    
    cv::Mat resizedImageBGR, resizedImageRGB,resizedImage;

    cv::resize(img, resizedImageBGR,
               cv::Size(target_width, target_height),
               cv::InterpolationFlags::INTER_CUBIC);

    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);

    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    // Float opencv mat to float ncnn

    ncnn::Mat in_pack3(resizedImage.cols, resizedImage.rows, 1, (void*)resizedImage.data, (size_t)4u * 3, 3);
    ncnn::Mat inRGB;
    ncnn::convert_packing(in_pack3, inRGB, 1);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);



    ncnn::Extractor ex = crackFinder.create_extractor() ;
    
    ex.input("img", inRGB);
    ex.extract("output_1", output_Scores);


    return 0;
}

int CrackFinder::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

//         fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0));

        if (has_kps)
        {
            cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return 0;
}
