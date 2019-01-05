#include "object_detection.hpp"
#include "kcftracker.hpp"
#include "HungarianAlg.h"
#include "kalman.hpp"
#include "ipm.hpp"

struct CAMERA
{
    float pitch;
    float yaw;
    float roll;
    float cx;
    float cy;

    CAMERA()
    {
        pitch = 2;
        yaw = 0;
        roll = 0;
        cx = 621;
        cy = 187;
    }
};


struct Object
{
    int id;
    int appearing_times;
    int missing_times;

    object_detection::BoundingBox predict_info;
    object_detection::BoundingBox old_info;
    double score;
    KCFTracker tracker;

    object_detection::BoundingBox predict_track_kalman_img;
    Kalman tracker2;

    std::vector<cv::Vec4f> pts_original;
    std::vector<cv::Vec4f> pts_filter;
    Kalman kalman_ipm;

    bool dump;
    bool left_or_right;


    Object()
    {
        missing_times = 0;
        score = 0.0;
        dump = true;
        left_or_right = false;
    }
};

class ObjectTracking
{
public:

    ObjectTracking(CAMERA camera_info = CAMERA(),
            int max_appearing_times = 3,
            int max_missing_detects_num = 10, //10
            int max_traces_num = 200,
            int max_filter_num = 30,
            int min_probability = 0.25,
            int min_object_area = 100,
            cv::Rect roi_in = cv::Rect(321, 200, 921, 374),
            cv::Rect roi_out = cv::Rect(0, 0, 400, 374))
    {
        max_traces_num_ = max_traces_num;
        max_filter_num_ = max_filter_num;
        max_appearing_times_ = max_appearing_times;
        max_missing_times_ = max_missing_detects_num;
        min_probability_ = min_probability;
        id_counter_ = 0;
        min_object_area_ = min_object_area;

        roi_in_ = roi_in;
        roi_out_ = roi_out;
        roi_stable_detects_ = cv::Rect(50, 50, 50, 20);
        ipm_transformer_.init_matrix(camera_info.pitch, camera_info.yaw, camera_info.roll,
                                     camera_info.cx, camera_info.cy, roi_in, roi_out);
    };

    std::vector<Object> doObjectTracking(
            const cv::Mat &im,
            std::vector<object_detection::BoundingBox> &detects
    );
    std::vector<Object> getObjects()
    {
        return objects_;
    }

    cv::Mat getImWarp()
    {
        return im_warp_;
    }

    IpmTransformer getIpmTransformer()
    {
        return ipm_transformer_;
    }

private:
    std::vector<Object> objects_;
    cv::Mat im_;
    cv::Mat im_warp_;

private:
    int max_appearing_times_;
    double min_probability_;
    int max_missing_times_;
    int min_object_area_;
    int id_counter_;

    int max_traces_num_;
    int max_filter_num_;
    cv::Rect roi_in_;
    cv::Rect roi_out_;
    cv::Rect roi_stable_detects_;

    IpmTransformer ipm_transformer_;


    double calculate_iou(cv::Rect &rect_a, const cv::Rect &rect_b);
    double calculate_color_iou(cv::Rect &rect_a, const cv::Rect &rect_b);
    cv::Mat calculate_histogram(int num_bin, cv::Mat im);
};

std::vector<Object> ObjectTracking::doObjectTracking(
        const cv::Mat &im,
        std::vector<object_detection::BoundingBox> &detects
)
{
    // 0. draw
//    cv::Mat im_draw = im.clone();
//    cv::Mat im_draw2 = im.clone();
//    cv::Mat im_draw3 = im.clone();

    // 1. do detect
    im_ = im.clone();
    cv::Mat im_new;
    cv::Rect im_rect(0, 0, im.cols, im.rows);
    cv::cvtColor(im_, im_new, cv::COLOR_BGR2HSV);
    cv::warpPerspective(im_, im_warp_, ipm_transformer_.get_img_to_ipm(), cv::Size(roi_out_.width, roi_out_.height));
    std::vector<cv::Mat> detects_hist;
    std::vector<cv::Mat> objects_hist;

    for (size_t i = 0; i < detects.size(); ++i)
    {
        detects[i].rect = im_rect & detects[i].rect;

//        detects[i].rect.x = std::max(0, std::min(im.cols, detects[i].rect.x));
//        detects[i].rect.y = std::max(0, std::min(im.rows, detects[i].rect.y));

//        detects[i].rect.width = std::max(0, std::min(im.cols - detects[i].rect.x, detects[i].rect.width));
//        detects[i].rect.height = std::max(0, std::min(im.rows - detects[i].rect.y, detects[i].rect.height));
//        cv::rectangle(im_draw2, detects[i].rect, cv::Scalar(0, 0, 255), 2);
//        imshow("detects", im_draw2);
//        cv::waitKey(0);
    }


    // 2. do tracking
    for (size_t i = 0; i < objects_.size(); ++i)
    {
        float score;
        //kcf tracking
        objects_[i].old_info.rect = objects_[i].tracker.update(im, score);
        objects_[i].score = score;

        std::vector<float> vals_ret = objects_[i].tracker2.do_predict();

        //kalman fiter
        objects_[i].predict_track_kalman_img.rect =
                cv::Rect(vals_ret[0] - vals_ret[2] / 2,
                         vals_ret[1] - vals_ret[3] / 2,
                         vals_ret[2],
                         vals_ret[3]);

        objects_[i].old_info.rect = objects_[i].old_info.rect & im_rect;
//        objects_[i].old_info.rect.x =
//                std::max(0, std::min(im.cols, objects_[i].old_info.rect.x));
//        objects_[i].old_info.rect.y =
//                std::max(0, std::min(im.rows, objects_[i].old_info.rect.y));
//        objects_[i].old_info.rect.width =
//                std::max(0, std::min(im.cols - objects_[i].old_info.rect.x, objects_[i].old_info.rect.width));
//        objects_[i].old_info.rect.height =
//                std::max(0, std::min(im.rows - objects_[i].old_info.rect.y, objects_[i].old_info.rect.height));

        objects_[i].predict_track_kalman_img.rect = objects_[i].predict_track_kalman_img.rect & im_rect;
//        objects_[i].predict_track_kalman_img.rect.x =
//                std::max(0, std::min(im.cols, objects_[i].predict_track_kalman_img.rect.x));
//        objects_[i].predict_track_kalman_img.rect.y =
//                std::max(0, std::min(im.rows, objects_[i].predict_track_kalman_img.rect.y));
//        objects_[i].predict_track_kalman_img.rect.width =
//                std::max(0, std::min(im.cols - objects_[i].predict_track_kalman_img.rect.x, objects_[i].predict_track_kalman_img.rect.width));
//        objects_[i].predict_track_kalman_img.rect.height =
//                std::max(0, std::min(im.rows - objects_[i].predict_track_kalman_img.rect.y, objects_[i].predict_track_kalman_img.rect.height));
    }

    //// calculate the cost func
    /*
    size_t num_detect = detects.size();
    size_t num_track = objects_.size();
    std::vector<std::vector<double> > costs(num_track, std::vector<double>(num_detect));

    for (size_t i = 0; i < num_track; ++i)
    {
        for (size_t j = 0; j < num_detect; ++j)
        {
            double iou = calculate_iou(objects_[i].old_info.rect, detects[j].rect);
            costs[i][j] = 1.0 - iou;
        }
    }*/

    // 3. calculate the cost func
    int num_tracks = objects_.size();
    int num_detects = detects.size();
    std::vector<std::vector<double> > costs(num_tracks,
                                            std::vector<double>(num_detects));
    std::vector<int> assignments;

    //QueryPerformanceCounter(&t1);

    //clock_gettime(CLOCK_REALTIME, &time2);
    //std::cout << "[info] time for step3: " << (time2.tv_sec - time1.tv_sec) * 1000 + (time2.tv_nsec - time1.tv_nsec) / 1000000 << "ms" << std::endl;

    //#pragma omp parallel for
    for (int i = 0; i < detects.size(); ++i)
    {
        //cv::Mat tmp = calculate_histogram(32, im_new(detects[i].rect));
        //detects[i].color_hist(tmp.size(), tmp.type());
        //tmp.copyTo(detects[i].color_hist);
        cv::Mat color_hist = calculate_histogram(32, im_new(detects[i].rect));
        detects_hist.push_back(color_hist);
    }

    //clock_gettime(CLOCK_REALTIME, &time2);
    //std::cout << "[info] time for step3: " << (time2.tv_sec - time1.tv_sec) * 1000 + (time2.tv_nsec - time1.tv_nsec) / 1000000 << "ms" << std::endl;

    //#pragma omp parallel for
    for (int i = 0; i < objects_.size(); ++i)
    {
        cv::Mat color_hist = calculate_histogram(32, im_new(objects_[i].old_info.rect));
        objects_hist.push_back(color_hist);
    }

    //clock_gettime(CLOCK_REALTIME, &time2);
    //std::cout << "[info] time for step3: " << (time2.tv_sec - time1.tv_sec) * 1000 + (time2.tv_nsec - time1.tv_nsec) / 1000000 << "ms" << std::endl;


    //#pragma omp parallel for
    for (size_t i = 0; i < num_tracks; ++i)
    {
    //#pragma omp parallel for
        for (size_t j = 0; j < num_detects; ++j)
        {
            // iou
            double iou = calculate_iou(objects_[i].old_info.rect, detects[j].rect);
            double score_similarity1 = cv::sum(cv::min(objects_hist[i],
                                                       detects_hist[j])).val[0];

            //double score_similarity1 = calculate_color_iou(objects_[i].old_info.rect,
            //	detects[j].rect);
            //std::cout << score_similarity1 << std::endl;
            double score_similarity2 = calculate_color_iou(objects_[i].predict_track_kalman_img.rect,
            	detects[j].rect);
            double score_area = std::min(objects_[i].old_info.rect.area(), detects[j].rect.area()) * 1.0
                                / std::max(objects_[i].old_info.rect.area(), detects[j].rect.area());


//            cv::rectangle(im_draw, objects_[i].old_info.rect, cv::Scalar(0, 255, 0), 2);
//            cv::rectangle(im_draw, detects[j].rect, cv::Scalar(0, 0, 255), 2);
//            imshow("sb", im_draw);
//            cv::waitKey(0);

            costs[i][j] = 1.0 - iou * score_similarity1 /** score_similarity2*/ * score_area;
            //costs[i][j] = 1.0 - (iou + score_similarity1 + score_area) / 3.0;
            //costs[i][j] = 1.0 - iou;
            //costs[i][j] = 1.0 - (iou * score_area * 1.0);
            //costs[i][j] = 1.0 - (iou + score_area) / 2.0;
            costs[i][j] = std::max(0.0, std::min(1.0, costs[i][j]));

            //if (costs[i][j] < 0)
            //{
            //	cv::Mat im_draw = im.clone();
            //	cv::rectangle(im_draw, _objects[i].old_track.rect, cv::Scalar(0, 255, 0), 2);
            //	cv::rectangle(im_draw, detects[j].rect, cv::Scalar(0, 0, 255), 2);
            //	imshow("sb", im_draw);
            //	cv::waitKey(0);
            //	i = i;
            //}

            //if (costs[i][j] > 0.8)
            //{
            //	costs[i][j] = 1000000000;
            //}
        }
    }

    //// use hungarian to solve it
    if (costs.size() > 0)
    {
        AssignmentProblemSolver solver;
        solver.Solve(costs, assignments, AssignmentProblemSolver::optimal);
    }

    //QueryPerformanceCounter(&t2);
    //std::cout << "step 3: " << (t2.QuadPart - t1.QuadPart) * 1000.0 / tc.QuadPart << std::endl;

//#ifdef __GNUC__
//    clock_gettime(CLOCK_REALTIME, &time2);
//    std::cout << "[info] time for step3: " << (time2.tv_sec - time1.tv_sec) * 1000 + (time2.tv_nsec - time1.tv_nsec) / 1000000 << "ms" << std::endl;
//#else
//    QueryPerformanceCounter(&t2);
//	std::cout << "[info] time for step3: " << (t2.QuadPart - t1.QuadPart) * 1000.0 / tc.QuadPart << std::endl;
//#endif

    // 4
    //// 4.1 if missing detect
    for (size_t i = 0; i < assignments.size(); ++i)
    {
        if (assignments[i] == -1 || costs[i][assignments[i]] > 0.9)
        {
            assignments[i] = -1;
            ++objects_[i].missing_times;
        }
        else
        {
            objects_[i].missing_times = 0;// std::max(0, _objects[i].missing_times - 1);
            ++objects_[i].appearing_times;
        }
    }

    for (size_t i = 0; i < objects_.size(); ++i)
    {
        if (objects_[i].missing_times >= max_missing_times_/* ||
			_objects[i].predict_track.rect.x < 30 ||
			_objects[i].predict_track.rect.x + _objects[i].predict_track.rect.width > im.cols - 30 ||
			_objects[i].predict_track.rect.y + _objects[i].predict_track.rect.height > 374 - 10 ||
			_objects[i].score < 0.3*/)
        {
            objects_.erase(objects_.begin() + i);
            objects_hist.erase(objects_hist.begin() + i);
            assignments.erase(assignments.begin() + i);
            --i;
        }
    }

    //4.2 if missing track, new
    for (size_t i = 0; i < detects.size(); ++i)
    {
        std::vector<int>::iterator iter = find(assignments.begin(),
                                               assignments.end(), i);
        if (iter == assignments.end())
        {
            if (detects[i].rect.x < roi_stable_detects_.x ||
                detects[i].rect.y < roi_stable_detects_.y ||
                detects[i].rect.width + detects[i].rect.x > im_.cols - roi_stable_detects_.width ||
                detects[i].rect.height + detects[i].rect.y > im_.rows - roi_stable_detects_.height ||
                detects[i].score < min_probability_ //||
                //detects[i].rect.area() < min_object_area_
                    )
            {
                continue;
            }

            // 4.2.1. new tracking object
            Object new_object;

            new_object.id = id_counter_++;
            new_object.missing_times = 0;
            new_object.appearing_times = 1;
            new_object.tracker.init(detects[i].rect, im);
            new_object.old_info = detects[i];
            new_object.predict_info = detects[i]; //needed?
            new_object.predict_track_kalman_img = detects[i];



            // 4.2.2. init kalman tracking
            std::vector<float> vals;
            vals.push_back(detects[i].rect.x + detects[i].rect.width / 2);
            vals.push_back(detects[i].rect.y + detects[i].rect.height / 2);
            vals.push_back(detects[i].rect.width);
            vals.push_back(detects[i].rect.height);
            new_object.tracker2.init(vals, 8);

            // 4.2.3. init ipm kalman filter
            cv::Rect rt = detects[i].rect;
            cv::Point2f pt;

            if (rt.x < im_.cols / 2)
            {
                new_object.left_or_right = false;
                pt = cv::Point2f(rt.x, rt.y + rt.height);

                //if (calculate_iou(cv::Rect(pt.x, pt.y))
                //{
                //}
            }
            else
            {
                new_object.left_or_right = true;
                pt = cv::Point2f(rt.x + rt.width, rt.y + rt.height);
            }

            vals.clear();
            pt = ipm_transformer_.pt_from_img_to_ipm(pt,
                                                     ipm_transformer_.get_img_to_ipm());

            vals.push_back(pt.x);
            vals.push_back(pt.y);
            new_object.kalman_ipm.init(vals, 4);

            new_object.pts_original.push_back(cv::Vec4f(pt.x, pt.y, 0, 0));
            new_object.pts_filter.push_back(cv::Vec4f(pt.x, pt.y, 0, 0));

            //omp_set_lock(&lock);
            objects_.push_back(new_object);
//            cv::rectangle(im_draw, new_object.old_info.rect, cv::Scalar(0, 255, 0), 2);
//            imshow("new objects", im_draw);
//            cv::waitKey(0);
            //omp_unset_lock(&lock);
        }
    }

    // 5 update
    for (int i = 0; i < assignments.size(); ++i)
    {
        // 5.1. update based detection
        if (assignments[i] != -1)
        {
            objects_[i].tracker.init(detects[assignments[i]].rect, im);
            //objects_[i].old_info = detects[assignments[i]];
            objects_[i].predict_info = detects[assignments[i]];
        }
        else
        {
            objects_[i].predict_info = objects_[i].old_info;
            //objects_[i].predict_info.label = 1;
        }

        // 5.2. update kalman filter
        std::vector<float> vals;

        vals.push_back(objects_[i].predict_info.rect.x + objects_[i].predict_info.rect.width / 2);
        vals.push_back(objects_[i].predict_info.rect.y + objects_[i].predict_info.rect.height / 2);
        vals.push_back(objects_[i].predict_info.rect.width);
        vals.push_back(objects_[i].predict_info.rect.height);
        objects_[i].tracker2.do_update(vals);

        if (assignments[i] == -1
            && calculate_iou(objects_[i].predict_track_kalman_img.rect,
                             objects_[i].predict_info.rect) < 0.2
            /*&& objects_[i].score < 0.1*/)
        {
            //_objects.erase(_objects.begin() + i);
            //assignments.erase(assignments.begin() + i);
            //--i;
            objects_[i].dump = false;
            continue;
        }

//        cv::rectangle(im_draw3, objects_[i].old_info.rect, cv::Scalar(255, 0, 0), 2);
//        imshow("update", im_draw3);
//        cv::waitKey(0);

        // 5.3. update ipm kalman filter
        vals.clear();
        cv::Rect rt = objects_[i].predict_info.rect;

        cv::Point2f pt;
        if (objects_[i].left_or_right == false)
        {
            objects_[i].left_or_right = false;
            pt = cv::Point2f(rt.x, rt.y + rt.height);
        }
        else
        {
            objects_[i].left_or_right = true;
            pt = cv::Point2f(rt.x + rt.width, rt.y + rt.height);
        }

        pt = ipm_transformer_.pt_from_img_to_ipm(pt,
                                                 ipm_transformer_.get_img_to_ipm());
        vals.push_back(pt.x);
        vals.push_back(pt.y);


        std::vector<float> vals_ret = objects_[i].kalman_ipm.do_predict();
        objects_[i].kalman_ipm.do_update(vals);

        objects_[i].pts_original.push_back(cv::Vec4f(pt.x, pt.y, 0, 0));
        objects_[i].pts_filter.push_back(cv::Vec4f(vals_ret[0],
                                                   vals_ret[1],
                                                   vals_ret[2],
                                                   vals_ret[3]));

        if (objects_[i].pts_original.size() > max_filter_num_)
        {
            objects_[i].pts_original.erase(objects_[i].pts_original.begin(),
                                           objects_[i].pts_original.end() - max_filter_num_);
        }
        if (objects_[i].pts_filter.size() > max_filter_num_)
        {
            objects_[i].pts_filter.erase(objects_[i].pts_filter.begin(),
                                         objects_[i].pts_filter.end() - max_filter_num_);
        }

        //// 5.4.

        //if (_objects[i].pts_filter.size() > 0)
        //{
        //	// calculate the next pos
        //	int idx = _objects[i].pts_filter.size() - 1;
        //	float pos_y = _objects[i].pts_filter[idx].val[1] + _objects[i].pts_filter[idx].val[3];

        //	if (pos_y > _roi_out.height)
        //	{
        //		_objects.erase(_objects.begin() + i);
        //		assignments.erase(assignments.begin() + i);
        //		std::cout << "out of the scope" << std::endl;
        //	}
        //}

    }

    for (int i = 0; i < objects_.size(); ++i)
    {
        if (objects_[i].dump == false)
        {
            objects_.erase(objects_.begin() + i);
            objects_hist.erase(objects_hist.begin() + i);
            --i;
        }
    }

    detects_hist.clear();
    objects_hist.clear();

    return objects_;
}

double ObjectTracking::calculate_iou(cv::Rect &rect_a, const cv::Rect &rect_b)
{
    double iou = 0.0;
    cv::Rect rt_inner;
    rt_inner.x = std::max(rect_a.x, rect_b.x);
    rt_inner.y = std::max(rect_a.y, rect_b.y);
    rt_inner.width = std::min(rect_a.x + rect_a.width, rect_b.x + rect_b.width);
    rt_inner.height = std::min(rect_a.y + rect_a.height, rect_b.y + rect_b.height);

    if (rt_inner.width > rt_inner.x && rt_inner.height > rt_inner.y)
    {
        iou = (rt_inner.width - rt_inner.x) * (rt_inner.height - rt_inner.y);
        iou = iou / (rect_a.area() + rect_b.area() - iou);
    }

    return iou;
}

cv::Mat ObjectTracking::calculate_histogram(int num_bin, cv::Mat im)
{
    int num = num_bin * im.channels();
    cv::Mat hist = cv::Mat::zeros(cv::Size(num, 1), CV_32FC1);

    int bin = ceil(256.0 / num_bin);
    uchar *hist_values = new uchar[256];
    for (int j = 0; j < 256; ++j)
    {
        hist_values[j] = j / bin;
    }

    int total_sz = im.rows * im.cols * im.channels();

    uchar *p_data = im.ptr<uchar>(0);
    for (int i = 0; i < total_sz;)
    {
        ++hist.at<float>(0, hist_values[p_data[i++]]);
        ++hist.at<float>(0, hist_values[p_data[i++]] + num_bin);
        ++hist.at<float>(0, hist_values[p_data[i++]] + num_bin * 2);
    }

    //for (int r = 0; r < im.rows; ++r)
    //{
    //	for (int c = 0; c < im.cols; ++c)
    //	{
    //		im.ptr
    //		hist.at<float>(0, im.at<cv::Vec3b>(r, c).val[0] / bin) += 1;
    //		hist.at<float>(0, im.at<cv::Vec3b>(r, c).val[1] / bin + num_bin) += 1;
    //		hist.at<float>(0, im.at<cv::Vec3b>(r, c).val[2] / bin + num_bin * 2) += 1;
    //	}
    //}

    //std::cout << hist << std::endl;

    hist = hist / cv::sum(hist).val[0];

    return hist;
}

double ObjectTracking::calculate_color_iou(cv::Rect &rect_a, const cv::Rect &rect_b)
{
    cv::Mat im_new;
    cv::cvtColor(im_, im_new, cv::COLOR_BGR2HSV);

    // 1. histogram
    cv::Mat im1 = im_new(rect_a).clone();
    cv::Mat hist1 = calculate_histogram(32, im1);
    cv::Mat im2 = im_new(rect_b).clone();
    cv::Mat hist2 = calculate_histogram(32, im2);

    // 2. similarity
    cv::Mat mat_inner = cv::min(hist1, hist2);
    //std::cout << cv::sum(hist1).val[0] << std::endl;
    //std::cout << cv::sum(hist2).val[0] << std::endl;
    //std::cout << cv::sum(mat_inner).val[0] << std::endl;

    return cv::sum(mat_inner).val[0];
}