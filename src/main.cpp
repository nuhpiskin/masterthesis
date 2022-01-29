#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>
#include <iostream>
#include "model.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"

void inference(cv::Mat &rgb, CrackFinder* g_scrfd, ncnn::Mutex& lock, cv::Mat& output_image)
{
    // scrfd
    {
        ncnn::MutexLockGuard g(lock);

        if (g_scrfd)
        {
          
            g_scrfd->detect(rgb, output_image);
        
        }
    }

}


void image_generator(std::string& path, CrackFinder* c_finder,ncnn::Mutex& lock)
{
    
    int length = 2;
    cv::Mat tmp;
    for (int i =0 ; i<length; i++)
    {
        cv::Mat src = cv::imread(path);
        src.copyTo(tmp);
        int64 start = cv::getTickCount();
        inference(src, c_finder, lock,tmp);
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;


        //std::cout << output_image << std::endl;
        cv::imwrite("/home/nmp/Desktop/projects/github/crackdetection/images/test1-res.jpg", tmp);
        
        
    }
    
    
}
void load_model_detector(CrackFinder* & c_finder, ncnn::Mutex& lock, bool use_gpu)
{
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete c_finder;
            c_finder = 0;
        }
        else
        {
            if (!c_finder)
                c_finder = new CrackFinder;
            // Here is ncnn g_scrfd load function
            c_finder->load(use_gpu);
        }
}


static CrackFinder* c_finder = 0;

static ncnn::Mutex lock;



void load_all_model()
{
    // Select  gpu usage
    bool use_gpu = false;
    //ncnn model load module
    load_model_detector(c_finder, lock, use_gpu);

}
int main()
{

    load_all_model();


    std::string path = {"/home/nmp/Desktop/projects/github/crackdetection/images/test1.jpg"};
 
    // Model inference function.
    image_generator(path, c_finder,lock);
    //image_generator(path, model, lock);
    return 0;
}