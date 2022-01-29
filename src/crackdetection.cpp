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

#include "model.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"

#include <iostream>
#include "cpu.h"


int CrackFinder::load( bool use_gpu)
{
    crackFinder.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    crackFinder.opt = ncnn::Option();

#if NCNN_VULKAN
    crackFinder.opt.use_vulkan_compute = use_gpu;
#endif

    crackFinder.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "/home/nmp/Desktop/projects/github/crackdetection/assets/models/segmentation_model_192x192_ONNX_simpifyed-sim-opt.param");
    sprintf(modelpath, "/home/nmp/Desktop/projects/github/crackdetection/assets/models/segmentation_model_192x192_ONNX_simpifyed-sim-opt.bin");

    crackFinder.load_param(parampath);
    crackFinder.load_model(modelpath);


    return 0;
}


void CrackFinder::DrawSegmentation(ncnn::Mat& results_segmentation,cv::Mat& output_image)
{
    ncnn::Mat in_pack3;
    ncnn::convert_packing(results_segmentation, in_pack3, 1);
    cv::Mat a(results_segmentation.h, results_segmentation.w, CV_32FC1);
    memcpy((uchar*)a.data, in_pack3.data, results_segmentation.w * results_segmentation.h * 1 * sizeof(float));
    
    cv::Mat R = cv::Mat::zeros(a.rows, a.cols,CV_8UC1);
    cv::Mat G = cv::Mat::zeros(a.rows, a.cols,CV_8UC1);
    cv::Mat B = cv::Mat::zeros(a.rows, a.cols,CV_8UC1);

    
    R.setTo(56,a>0.5); // USEFUL Script
    G.setTo(34,a>0.5); // USEFUL Script
    B.setTo(132,a>0.5); // USEFUL Script

    cv::Mat outRGB;
    cv::Mat in[] = {B, G, R};
    cv::merge(in, 3, outRGB);

    cv::resize(outRGB,outRGB,cv::Size(output_image.cols, output_image.rows));

    double alpha = 0.5; double beta; 
    beta = ( 1.0 - alpha );

    cv::addWeighted( output_image, alpha, outRGB, beta, 0.0, output_image);

   
}
int CrackFinder::detect(const cv::Mat& rgb,cv::Mat& output_image)
{
    int target_width =192; int target_height =192;
    
    cv::Mat resizedImageBGR, resizedImageRGB,resizedImage;

    cv::resize(rgb, resizedImageBGR,
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


    ncnn::Extractor ex = crackFinder.create_extractor() ;
    
    ncnn::Mat results_segmentation;
    ex.input("img", inRGB);
    ex.extract("output_1", results_segmentation);
    DrawSegmentation(results_segmentation, output_image);
    return 0;
}
