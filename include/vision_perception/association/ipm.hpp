#ifndef ADAS_MODULE_IPM_H
#define ADAS_MODULE_IPM_H

#include <opencv2/opencv.hpp>

class IpmTransformer
{
public:
	cv::Point2f pt_from_img_to_ipm(cv::Point2f src_pt, cv::Mat trans)
	{
		cv::Point2f dst_pt;

		cv::Mat pt_m = (cv::Mat_<float>(3, 1) << src_pt.x, src_pt.y, 1);
		trans.convertTo(trans, CV_32F);
		//std::cout << trans;

		cv::Mat pt_m_new = trans * pt_m;
		dst_pt.x = pt_m_new.at<float>(0, 0) / pt_m_new.at<float>(2, 0);
		dst_pt.y = pt_m_new.at<float>(1, 0) / pt_m_new.at<float>(2, 0);

		return dst_pt;
	}

	int init_matrix(float pitch, float yaw, float roll,
		float cx, float cy, cv::Rect roi_in, cv::Rect roi_out)
	{
		// 1. 矩阵准备
		float c1 = cos(pitch * CV_PI / 180);
		float s1 = sin(pitch * CV_PI / 180);
		cv::Mat pitch_m = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, c1, s1, 0, -s1, c1);

		

		float c2 = cos(yaw * CV_PI / 180);
		float s2 = sin(yaw * CV_PI / 180);
		cv::Mat yaw_m = (cv::Mat_<float>(3, 3) << c2, s2, 0, -s2, c2, 0, 0, 0, 1);

		cv::Mat camera_to_world_m = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
		cv::Mat image_to_camera_m = (cv::Mat_<float>(3, 3) << 1 / 700.0, 0, -cx / 700.0, 
			0, 1 / 700.0, -cy / 700.0, 
			0, 0, 1);

		cv::Mat image_to_world_m = yaw_m * pitch_m * camera_to_world_m * image_to_camera_m;

		// 2. 通过映射点计算变换矩阵
		cv::Point2f src_pts[4];
		cv::Point2f dst_pts[4];

		src_pts[0] = cv::Point2f(roi_in.x, roi_in.y);
		src_pts[1] = cv::Point2f(roi_in.width, roi_in.y);
		src_pts[2] = cv::Point2f(roi_in.width, roi_in.height);
		src_pts[3] = cv::Point2f(roi_in.x, roi_in.height);

		float min_x = INT_MAX;
		float max_x = INT_MIN;
		float min_y = INT_MAX;
		float max_y = INT_MIN;
		for (int i = 0; i < 4; ++i)
		{
			dst_pts[i] = pt_from_img_to_ipm(src_pts[i], image_to_world_m);

			//std::cout << image_to_world_m;

			min_x = std::min(min_x, dst_pts[i].x);
			max_x = std::max(max_x, dst_pts[i].x);
			min_y = std::min(min_y, dst_pts[i].y);
			max_y = std::max(max_y, dst_pts[i].y);
		}

		for (int i = 0; i < 4; ++i)
		{
			dst_pts[i].x = (dst_pts[i].x - min_x) / (max_x - min_x) * roi_out.width;
			dst_pts[i].y = (dst_pts[i].y - min_y) / (max_y - min_y) * roi_out.height;
		}

		_img_to_ipm = cv::getPerspectiveTransform(src_pts, dst_pts);
		cv::invert(_img_to_ipm, _ipm_to_img);

		return 0;
	}

	cv::Mat get_img_to_ipm()
	{
		return _img_to_ipm;
	}
	cv::Mat get_ipm_to_img()
	{
		return _ipm_to_img;
	}

private:
	cv::Mat _img_to_ipm;
	cv::Mat _ipm_to_img;
};


#endif
