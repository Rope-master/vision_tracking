//
// Created by grayson on 18-1-9.
//

#ifndef VISION_DETECTION_YOLO_DETECTION_HPP
#define VISION_DETECTION_YOLO_DETECTION_HPP

#include "darknet/darknet.h"
#include <string>
#include <opencv2/core/core.hpp>
#include <iostream>

namespace yolo_detection
{
    class YOLODetector
    {
    public:
        YOLODetector() = default;

        YOLODetector(const std::string &model_file,
                     const std::string &weights_file,
                     const std::string &labels_file,
                     double confidence_threshold);

        detection * detect(const cv::Mat &img, int &nboxes);

        network *getNetwork();

        bool isInitialized();

    private:
        image ipl_to_image(IplImage *src);

    private:
        bool initialized_ = false;
        double confidence_threshold_;
        network *net_;
    };

    YOLODetector::YOLODetector(const std::string &model_file, const std::string &weights_file,
                               const std::string &labels_file, double confidence_threshold)
    {
        //    char **names = get_labels(const_cast<char *>(labels_file.c_str()));
        //    image **alphabet = load_alphabet();
        net_ = load_network((char *)model_file.c_str(), (char *)weights_file.c_str(), 0);
        set_batch_network(net_, 1);
        initialized_ = true;
        confidence_threshold_ = confidence_threshold;
    }

    bool YOLODetector::isInitialized()
    {
        return initialized_;
    }

    detection *YOLODetector::detect(const cv::Mat &img, int &nboxes)
    {
        IplImage ipl_img = img;
        image im = ipl_to_image(&ipl_img);
        image sized = letterbox_image(im, net_->w, net_->h);
        layer l = net_->layers[net_->n - 1];

        float nms = .45, hier_thresh = .5;
        float *X = sized.data;
        network_predict(net_, X);
        detection *dets = get_network_boxes(net_, im.w, im.h, confidence_threshold_, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms)
        {
            do_nms_sort(dets, nboxes, l.classes, nms);
        }
        free_image(im);
        free_image(sized);
        return dets;
    }

    image YOLODetector::ipl_to_image(IplImage *src)
    {
        int h = src->height;
        int w = src->width;
        int c = src->nChannels;
        int step = src->widthStep;

        image out = make_image(w, h, c);
        unsigned char *data = (unsigned char *)src->imageData;

        int i, j, k;

        for (i = 0; i < h; ++i)
        {
            for (k = 0; k < c; ++k)
            {
                for (j = 0; j < w; ++j)
                {
                    out.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.;
                }
            }
        }

        return out;
    }

    network *YOLODetector::getNetwork()
    {
        return net_;
    }
}
#endif //VISION_DETECTION_YOLO_DETECTION_HPP
