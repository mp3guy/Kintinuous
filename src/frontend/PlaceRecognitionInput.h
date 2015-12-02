/*
 * This file is part of Kintinuous.
 *
 * Copyright (C) 2015 The National University of Ireland Maynooth and 
 * Massachusetts Institute of Technology
 *
 * The use of the code within this file and all code within files that 
 * make up the software that is Kintinuous is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.cs.nuim.ie/research/vision/data/kintinuous/code.php> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email commercialisation@nuim.ie.
 */

#ifndef PLACERECOGNITIONINPUT_H_
#define PLACERECOGNITIONINPUT_H_

#include "../utils/ConfigArgs.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <zlib.h>

class PlaceRecognitionInput
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PlaceRecognitionInput(unsigned char * rgbImage,
                              int imageSize,
                              unsigned short * depthMap,
                              int depthSize,
                              bool isCompressed,
                              uint64_t utime,
                              uint64_t lagTime,
                              Eigen::Vector3f & trans,
                              Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rotation)
         : rgbImage(rgbImage),
           imageSize(imageSize),
           depthMap(depthMap),
           depthSize(depthSize),
           isCompressed(isCompressed),
           originallyCompressed(isCompressed),
           utime(utime),
           lagTime(lagTime),
           trans(trans),
           rotation(rotation),
           encodedImage(0),
           depth_compress_buf_size(Resolution::get().numPixels() * sizeof(int16_t) * 4),
           depth_compress_buf(0)
        {}

        PlaceRecognitionInput()
        : rgbImage(0),
          imageSize(0),
          depthMap(0),
          depthSize(0),
          isCompressed(false),
          originallyCompressed(false),
          utime(0),
          lagTime(0),
          encodedImage(0),
          depth_compress_buf_size(Resolution::get().numPixels() * sizeof(int16_t) * 4),
          depth_compress_buf(0)
        {
            dump();
        }

        void compress()
        {
            assert(!isCompressed);

            if(!depth_compress_buf)
            {
                depth_compress_buf = new uint8_t[depth_compress_buf_size];
            }

            unsigned long compressed_size = depth_compress_buf_size;
            boost::thread_group threads;

            threads.add_thread(new boost::thread(compress2,
                                                 depth_compress_buf,
                                                 &compressed_size,
                                                 (const Bytef*)depthMap,
                                                 depthSize,
                                                 Z_BEST_SPEED));

            threads.add_thread(new boost::thread(boost::bind(&PlaceRecognitionInput::encodeJpeg,
                                                             this,
                                                             (cv::Vec<unsigned char, 3> *)rgbImage)));

            threads.join_all();

            uint8_t * tmpDepthBuffer = new uint8_t[compressed_size];
            memcpy(tmpDepthBuffer, depth_compress_buf, compressed_size);
            depthSize = compressed_size;

            delete [] depth_compress_buf;
            depth_compress_buf = 0;

            delete [] depthMap;
            depthMap = (unsigned short *)tmpDepthBuffer;

            uint8_t * tmpImgBuffer = new uint8_t[encodedImage->width];
            memcpy(tmpImgBuffer, encodedImage->data.ptr, encodedImage->width);
            imageSize = encodedImage->width;

            cvReleaseMat(&encodedImage);
            encodedImage = 0;

            delete [] rgbImage;
            rgbImage = (unsigned char *)tmpImgBuffer;

            isCompressed = true;
        }

        void decompressImgTo(uchar * target)
        {
            assert(isCompressed);
            CvMat tempMat = cvMat(1, imageSize, CV_8UC1, (void *)rgbImage);
            IplImage * deCompImage = cvDecodeImage(&tempMat);

            if(ConfigArgs::get().flipColors && originallyCompressed)
            {
                cvCvtColor(deCompImage, deCompImage, CV_BGR2RGB);
            }

            memcpy(target, deCompImage->imageData, Resolution::get().numPixels() * 3);
            cvReleaseImage(&deCompImage);
        }

        void decompressDepthTo(uchar * target)
        {
            assert(isCompressed);
            unsigned long decompLength = Resolution::get().numPixels() * 2;
            uncompress(target, (unsigned long *)&decompLength, (const Bytef *)depthMap, depthSize);
        }

        void dump()
        {
            if(rgbImage)
            {
                delete [] rgbImage;
            }

            rgbImage = 0;

            imageSize = 0;

            if(depthMap)
            {
                delete [] depthMap;
            }

            depthMap = 0;

            depthSize = 0;

            isCompressed = false;

            utime = 0;

            lagTime = 0;

            trans(0) = 0;
            trans(1) = 0;
            trans(2) = 0;

            rotation.setIdentity();

            if(encodedImage)
            {
                cvReleaseMat(&encodedImage);
            }

            encodedImage = 0;

            if(depth_compress_buf)
            {
                delete [] depth_compress_buf;
            }

            depth_compress_buf = 0;
        }

        virtual ~PlaceRecognitionInput()
        {
            dump();
        }

        unsigned char * rgbImage;
        int imageSize;
        unsigned short * depthMap;
        int depthSize;
        bool isCompressed;
        const bool originallyCompressed;
        uint64_t utime;
        uint64_t lagTime;
        Eigen::Vector3f trans;
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation;
        std::vector<float> descriptor;
        std::vector<cv::KeyPoint> keyPoints;

    private:
        void encodeJpeg(cv::Vec<unsigned char, 3> * rgb_data)
        {
            cv::Mat3b rgb(Resolution::get().rows(), Resolution::get().cols(), rgb_data, Resolution::get().width() * 3);

            IplImage * img = new IplImage(rgb);

            int jpeg_params[] = {CV_IMWRITE_JPEG_QUALITY, 90, 0};

            if(encodedImage)
            {
                cvReleaseMat(&encodedImage);
            }

            encodedImage = cvEncodeImage(".jpg", img, jpeg_params);

            delete img;
        }

        CvMat * encodedImage;
        const int depth_compress_buf_size;
        uint8_t * depth_compress_buf;
};

#endif /* PLACERECOGNITIONINPUT_H_ */
