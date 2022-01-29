#ifndef SCRFD_H
#define SCRFD_H

#include <opencv2/core/core.hpp>

#include <net.h>



class CrackFinder
{
public:
    int load(bool use_gpu = false);
    int detect(const cv::Mat& rgb, cv::Mat& output_image);
    void DrawSegmentation(ncnn::Mat& results_segmentation,cv::Mat& output_image);

private:
    ncnn::Net crackFinder;
};

#endif // SCRFD_H