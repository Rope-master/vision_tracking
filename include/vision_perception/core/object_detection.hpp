#ifndef VISION_DETECTION_OBJECT_DETECTION_H
#define VISION_DETECTION_OBJECT_DETECTION_H
#define SSD
#ifdef SSD
#include "ssd_detection.hpp"
#elif defined(YOLOV3)
#include "yolo_detection.hpp"
#endif

namespace object_detection
{
    struct BoundingBox
    {
        int label;
        double score;
        cv::Rect rect;
    };

    struct DetectionParameter
    {
        std::string model_file;
        std::string weights_file;
        std::string labels_file;
    };

    class Detector
    {
      public:
        Detector() = default;

        void doInit(std::string model_type, const DetectionParameter &para, double confidence_threshold);

        std::vector<BoundingBox> doDetectionOnline(const cv::Mat &img);

        int doDetectionOffline(const std::string &path_name);

      private:
#ifdef SSD
        std::vector<BoundingBox> parseSSDResult(caffe::Blob<float> *result_blob, const cv::Size &size);
#elif defined(YOLOV3)
        std::vector<BoundingBox> parseYOLOResult(detection *dets, int nboxes, const cv::Size &size);
#endif

      private:
        std::string model_type_;
        std::vector<BoundingBox> detections_;
        double confidence_threshold_;
#ifdef SSD
        ssd_detection::SSDDetector ssd_detector_;
#elif defined(YOLOV3)
        yolo_detection::YOLODetector yolo_detector_;
#endif
    };

    void Detector::doInit(const std::string model_type, const DetectionParameter &para, double confidence_threshold)
    {
        model_type_ = model_type;
        confidence_threshold_ = confidence_threshold;

#ifdef SSD
        ssd_detector_ = ssd_detection::SSDDetector(para.model_file, para.weights_file);
#elif defined(YOLOV3)
        yolo_detector_ = yolo_detection::YOLODetector(para.model_file, para.weights_file, para.labels_file,
                                                      confidence_threshold_);
#endif
    }

    std::vector<BoundingBox> Detector::doDetectionOnline(const cv::Mat &img)
    {
        //// use cnn to do it
#ifdef SSD
        {
            if (!ssd_detector_.isInitialized())
            {
                std::cout << "Initialize SSD model first!" << std::endl;
            }
            detections_ = parseSSDResult(ssd_detector_.detect(img), img.size());
        }

#elif defined(YOLOV3)
        {
            if (!yolo_detector_.isInitialized())
            {
                std::cout << "Initialize YOLO model first!" << std::endl;
            }

            int nboxes = 0;
            detection *dets = yolo_detector_.detect(img, nboxes);
            detections_ = parseYOLOResult(dets, nboxes, img.size());
            free_detections(dets, nboxes);
        }
#endif
        return detections_;
    }

    int Detector::doDetectionOffline(const std::string &path_name)
    {
        //// load txt file

        return 0;
    }

#ifdef SSD
    std::vector<BoundingBox> Detector::parseSSDResult(caffe::Blob<float> *result_blob, const cv::Size &size)
    {
        std::vector<BoundingBox> detections;

        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        const int num_det = result_blob->height();
        const float *result = result_blob->cpu_data();
        for (int k = 0; k < num_det; ++k)
        {
            double score = static_cast<double>(result[2]);

            if (result[0] == -1 || score < confidence_threshold_)
            {
                // Skip invalid detection and confidence less than threshold detection
                result += 7;
                continue;
            }

            int label = static_cast<int>(result[1]);
            int x_min = static_cast<int>(result[3] * size.width);
            int y_min = static_cast<int>(result[4] * size.height);
            int x_max = static_cast<int>(result[5] * size.width);
            int y_max = static_cast<int>(result[6] * size.height);

            cv::Rect rect(x_min, y_min, x_max - x_min, y_max - y_min);
            BoundingBox detection{.label = label, .score = score, .rect = rect};
            detections.push_back(detection);
            result += 7;
        }
        return detections;
    }
#elif defined(YOLOV3)
    std::vector<BoundingBox> Detector::parseYOLOResult(detection *dets, int nboxes, const cv::Size &size)
    {
        std::vector<BoundingBox> detections;

        for (int i = 0; i < nboxes; ++i)
        {
            box b = dets[i].bbox;
            int classes = sizeof(dets[i].prob) / sizeof(dets[i].prob[0]);
            for (int j = 0; j < classes; ++j)
            {
                if (dets[i].prob[j] > confidence_threshold_)
                {
                    int x = static_cast<int>((b.x - b.w / 2.) * size.width);
                    int y = static_cast<int>((b.y - b.h / 2.) * size.height);
                    int width = static_cast<int>(b.w * size.width);
                    int height = static_cast<int>(b.h * size.height);
                    cv::Rect rect(x, y, width, height);
                    BoundingBox detection{.label = j, .score = dets[i].prob[j], .rect = rect};
                    detections.push_back(detection);
                }
            }
        }
        return detections;
    }
#endif
}

#endif