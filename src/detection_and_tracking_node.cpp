#include "object_detection.hpp"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include <time.h>
#include "object_tracking.hpp"
#include "RectArray.h"

class DetectionAndTracking
{
public:
    DetectionAndTracking() = default;

    int doInit();
    int doInit_offline();
    int doInit_test_rectarray();
    void doDetectTrackingForImg(cv::Mat &img);

    std::vector<object_detection::BoundingBox> doDetectForCvMat(const cv::Mat &img);

    cv_bridge::CvImagePtr cv_image_;

private:
    void doDetectForSensorMsg(const sensor_msgs::ImageConstPtr &_img);
    void doTestRectArray(const vision_perception::RectArrayConstPtr &objects);

private:
    ros::NodeHandle n_;
    object_detection::Detector detector_;
    ObjectTracking tracker_;
    std::string model_type_;
    std::string config_file_, model_file_, weights_file_, result_file_, labels_file_;
    double confidence_threshold_;
    ros::Subscriber sub_;
    ros::Publisher  pub_;
};

int DetectionAndTracking::doInit()
{
    sub_ = n_.subscribe("/middle/pylon_camera_node/image_raw", 1, &DetectionAndTracking::doDetectForSensorMsg,
                        this);

    pub_ = n_.advertise<vision_perception::RectArray>("objects", 1000);

    ros::param::get("~config_file", config_file_);
    YAML::Node config = YAML::LoadFile(config_file_);
    confidence_threshold_ = config["confidence_threshold"].as<double>();

#ifdef SSD
    model_type_ = "ssd";
#elif defined(YOLO)
    model_type_ = "yolo";
#endif

    ros::param::get("~" + model_type_ + "_model_file", model_file_);
    ros::param::get("~" + model_type_ + "_weights_file", weights_file_);
    ros::param::get("~" + model_type_ + "_labels_file", labels_file_);
    ros::param::get("~" + model_type_ + "_result_file", result_file_);

    object_detection::DetectionParameter detection_parameter{.model_file=model_file_, .weights_file=weights_file_, .labels_file=labels_file_};
    detector_.doInit(model_type_, detection_parameter, confidence_threshold_);
    ROS_INFO("Using %s model", model_type_.c_str());
    return 0;
}

int DetectionAndTracking::doInit_test_rectarray()
{
    sub_ = n_.subscribe("objects", 1, &DetectionAndTracking::doTestRectArray,
                        this);
    return 0;
}

void DetectionAndTracking::doTestRectArray(const vision_perception::RectArrayConstPtr &objects)
{
    size_t x_size = objects->x.size();
    size_t y_size = objects->y.size();
    size_t width_size = objects->width.size();
    size_t height_size = objects->height.size();
    ROS_INFO("x_size:%i, y_size:%i, width_size:%i, height_size:%i", x_size, y_size, width_size, height_size);
}

int DetectionAndTracking::doInit_offline()
{

#ifdef SSD
    model_type_ = "ssd";
#elif defined(YOLO)
    model_type_ = "yolo";
#endif

    confidence_threshold_ = 0.15;
    model_file_ = "/home/xd/catkin_ws/src/vision_perception/models/SSD/tianjin/deploy_300x300.prototxt";
    weights_file_ = "/home/xd/catkin_ws/src/vision_perception/models/SSD/tianjin/VGG_tianjin_selected_SSD_300x300_iter_120000.caffemodel";
    labels_file_ = "/home/xd/catkin_ws/src/vision_perception/models/SSD/tianjin/labelmap_tianjin.prototxt";

    object_detection::DetectionParameter detection_parameter{.model_file=model_file_, .weights_file=weights_file_, .labels_file=labels_file_};
    detector_.doInit(model_type_, detection_parameter, confidence_threshold_);

    return 0;
}

std::vector<object_detection::BoundingBox> DetectionAndTracking::doDetectForCvMat(const cv::Mat &img)
{
    std::vector<object_detection::BoundingBox> result;
    result = detector_.doDetectionOnline(img);


//    std::ofstream result_ofstream;
//    result_ofstream.open(result_file_);
//    for (object_detection::BoundingBox bounding_box:result)
//    {
//        result_ofstream << bounding_box.label << " " << bounding_box.score << " " << bounding_box.rect << std::endl;
//    }
//    result_ofstream.close();
//
//    ROS_INFO("Detection result saved in %s", result_file_.c_str());

    return result;
}

void DetectionAndTracking::doDetectTrackingForImg(cv::Mat &img)
{
    //cv::Mat img_new = img.clone();
    std::vector<object_detection::BoundingBox> result = doDetectForCvMat(img);

    // with tracking
    for (Object object:tracker_.doObjectTracking(img, result))
    {
        if (object.old_info.label != 5 && object.appearing_times > 2 && object.missing_times < 4)    //ignore class "others"
        {
            cv::rectangle(img, object.old_info.rect, cv::Scalar(0, 0, 255), 2);
            string id_str = "id=" + std::to_string(object.id);
            cv::putText(img, id_str, object.old_info.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 255, 255));
        }
    }

    cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
    cv::imshow("image", img);
    cv::waitKey(1);
}

void DetectionAndTracking::doDetectForSensorMsg(const sensor_msgs::ImageConstPtr& _img)
{
    struct timespec start_time, finish_time;
    clock_gettime(CLOCK_REALTIME, &start_time);

    try 
    {
        cv_image_ = cv_bridge::toCvCopy(_img, sensor_msgs::image_encodings::BGR8);//BGR8
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat img = cv_image_->image.clone();
    cv::resize(img, img, cv::Size(960, 600), 0, 0, cv::INTER_LINEAR);
    std::vector<object_detection::BoundingBox> result = doDetectForCvMat(img);
    vision_perception::RectArray pub_msg;

    // with tracking
    for (Object object:tracker_.doObjectTracking(img, result))
    {
        if (object.old_info.label != 5)    //ignore class "others"
        {
            cv::rectangle(img, object.old_info.rect, cv::Scalar(0, 0, 255), 2);
            pub_msg.x.push_back(object.old_info.rect.x);
            pub_msg.y.push_back(object.old_info.rect.y);
            pub_msg.width.push_back(object.old_info.rect.width);
            pub_msg.height.push_back(object.old_info.rect.height);
            pub_.publish(pub_msg);
            string id_str = "id=" + std::to_string(object.id);
            cv::putText(img, id_str, object.old_info.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 255, 255));
        }
    }

    // without tracking
//    for (object_detection::BoundingBox bounding_box:result)
//    {
//        if (bounding_box.label != 5)    //ignore class "others"
//        {
//            cv::rectangle(img, bounding_box.rect, cv::Scalar(0, 0, 255), 2);
//        }
//    }

    cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
    cv::imshow("image", img);
    cv::waitKey(1);

    clock_gettime(CLOCK_REALTIME, &finish_time);
    ROS_INFO("%f fps",
             1 / ((finish_time.tv_sec + finish_time.tv_nsec * 1e-9) - (start_time.tv_sec + start_time.tv_nsec * 1e-9)));
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detection_and_tracking");
    DetectionAndTracking detection_and_tracking;
    detection_and_tracking.doInit();
//    detection_and_tracking.doInit_test_rectarray();
    ros::spin();

    // detection offline
//    DetectionAndTracking detection_and_tracking;
//    detection_and_tracking.doInit_offline();
//    ifstream image_list("/home/xd/Downloads/2018-01-10-15-02-01/name.txt");//("/home/xd/Downloads/2018-01-11-20-14-22/name.txt");//"/home/xd/Data/2018-01-10-15-02-01/name.txt"
//    string s;
//    ofstream res_list("/home/xd/Downloads/2018-01-10-15-02-01/res.txt");
//    cv::VideoWriter video("/home/xd/Downloads/2018-01-10-15-02-01/res.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, cvSize(960, 600));
//    while (getline(image_list, s))
//    {
//        cv::Mat img = cv::imread("/home/xd/Downloads/2018-01-10-15-02-01/" + s, cv::IMREAD_UNCHANGED);
//        cv::resize(img, img, cv::Size(960, 600), 0, 0, cv::INTER_LINEAR);
//        detection_and_tracking.doDetectTrackingForImg(img);
//        video.write(img);
//        res_list << s << '\n';


//        std::vector<object_detection::BoundingBox> result = detection_and_tracking.doDetectForCvMat(img);
//
//        ofstream result_file("home/xd/catkin_ws/src/vision_perception/tracking_res/" + s.substr(0, s.find('.')) + ".txt");
//        for (object_detection::BoundingBox bounding_box:result)
//        {
//            result_file << bounding_box.label << " ";
//            result_file << bounding_box.rect.tl().x << " " << bounding_box.rect.tl().y << " ";
//            result_file << bounding_box.rect.width << " " << bounding_box.rect.height << " ";
//            result_file << bounding_box.score << '\n';
//        }
//        result_file.close();
//    }
//    image_list.close();
//    res_list.close();

    // test
//    cv::Mat img = cv::imread("/media/grayson/DATA/ssd_img_pcd/1515565757084795.jpg", cv::IMREAD_UNCHANGED);
//    std::vector<object_detection::BoundingBox> result = detection_and_tracking.doDetectForCvMat(img);
//
//    for (object_detection::BoundingBox bounding_box:result)
//    {
//        cv::rectangle(img, bounding_box.rect, cv::Scalar(255, 0, 0), 2);
//    }
//    cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
//    cv::imshow("image", img);
//    cv::waitKey(0);
//    cv::destroyWindow("image");

    return 0;
}
