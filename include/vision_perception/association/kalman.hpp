#ifndef KALMAN_H
#define KALMAN_H
#include <opencv2/opencv.hpp>

class Kalman
{

public:

	//void init(cv::Vec4f vec)
	//{
	//	_vecs.push_back(vec);
	//}

	//cv::Vec4f do_filter(cv::Vec4f vec)
	//{
	//	int count_vec = _vecs.size();
	//	int count = std::max(0, std::min(10, count_vec));

	//	cv::Vec4f sum = cv::Vec4f(0, 0, 0, 0);
	//	for (int i = count_vec - 1; count_vec > 0 && i >= count_vec - count; --i)
	//	{
	//		sum += _vecs[i];
	//	}

	//	sum += (0 + 1) * vec;
	//	sum /= (count + 0 + 1);

	//	_vecs.push_back(vec);

	//	if (_vecs.size() > 10)
	//	{
	//		_vecs.pop_front();
	//	}

	//	return sum;
	//}


    void init(std::vector<float> vals, int mode = 4)
    {
		_mode = mode;

		if (_mode == 4)
		{
			_measure_num = 2;
			_state_num = 4;

			_kalman.init(_state_num, _measure_num, 0);

			_kalman.transitionMatrix = (cv::Mat_<float>(_state_num, _state_num) <<
				1, 0, 1, 0, 
				0, 1, 0, 1, 
				0, 0, 1, 0, 
				0, 0, 0, 1);

			setIdentity(_kalman.measurementMatrix, cv::Scalar::all(1));
			setIdentity(_kalman.processNoiseCov, cv::Scalar::all(1e-5));
			setIdentity(_kalman.measurementNoiseCov, cv::Scalar::all(1));
			setIdentity(_kalman.errorCovPost, cv::Scalar::all(1));

			_kalman.statePost.at<float>(0, 0) = vals[0];
			_kalman.statePost.at<float>(1, 0) = vals[1];
			_kalman.statePost.at<float>(2, 0) = 0;
			_kalman.statePost.at<float>(3, 0) = 0;
		}
		else if (_mode == 8)
		{
			_measure_num = 4;
			_state_num = 8;

			_kalman.init(_state_num, _measure_num, 0);

			_kalman.transitionMatrix = (cv::Mat_<float>(_state_num, _state_num) <<
				1, 0, 0, 0, 1, 0, 0, 0,
				0, 1, 0, 0, 0, 1, 0, 0,
				0, 0, 1, 0, 0, 0, 1, 0,
				0, 0, 0, 1, 0, 0, 0, 1,
				0, 0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 0, 1, 0, 0,
				0, 0, 0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 0, 0, 1);

			setIdentity(_kalman.measurementMatrix, cv::Scalar::all(1));
			setIdentity(_kalman.processNoiseCov, cv::Scalar::all(1e-3));
			setIdentity(_kalman.measurementNoiseCov, cv::Scalar::all(1e-2));
			setIdentity(_kalman.errorCovPost, cv::Scalar::all(1));

			// wh��״̬������С���۲���������
			//_kalman.measurementNoiseCov.at<float>(2, 2) = 1;
			//_kalman.measurementNoiseCov.at<float>(3, 3) = 1;

			//_kalman.processNoiseCov.at<float>(2, 2) = 1;
			//_kalman.processNoiseCov.at<float>(3, 3) = 1;
			//_kalman.processNoiseCov.at<float>(6, 6) = 1;
			//_kalman.processNoiseCov.at<float>(7, 7) = 1;

			//_kalman.measurementNoiseCov.at<float>(2, 2) = 1e-3;
			//_kalman.measurementNoiseCov.at<float>(3, 3) = 1e-3;

			//_kalman.processNoiseCov.at<float>(2, 2) = 1;
			//_kalman.processNoiseCov.at<float>(3, 3) = 1;
			//_kalman.processNoiseCov.at<float>(6, 6) = 1;
			//_kalman.processNoiseCov.at<float>(7, 7) = 1;

			_kalman.statePost.at<float>(0, 0) = vals[0];
			_kalman.statePost.at<float>(1, 0) = vals[1];
			_kalman.statePost.at<float>(2, 0) = vals[2];
			_kalman.statePost.at<float>(3, 0) = vals[3];
			_kalman.statePost.at<float>(4, 0) = 0;
			_kalman.statePost.at<float>(5, 0) = 0;
			_kalman.statePost.at<float>(6, 0) = 0;
			_kalman.statePost.at<float>(7, 0) = 0;
		}
        
    }

	std::vector<float> do_predict()
	{
		std::vector<float> ret;
		cv::Mat state_m = _kalman.predict();

		for (int i = 0; i < state_m.rows; ++i)
		{
			ret.push_back(state_m.at<float>(i, 0));
		}

		return ret;
	}

	int do_update(std::vector<float> vals)
    {
		if (_mode == 4)
		{
			// 2. �����µĹ۲�ֵ���¹۲ⷽ��
			cv::Mat measure_m = (cv::Mat_<float>(_measure_num, 1) << vals[0], vals[1]);
			_kalman.correct(measure_m);
		}
		else if (_mode == 8)
		{
			// 2. �����µĹ۲�ֵ���¹۲ⷽ��
			cv::Mat measure_m = (cv::Mat_<float>(_measure_num, 1) << vals[0], vals[1], vals[2], vals[3]);
			_kalman.correct(measure_m);
		}
		
		return 0;
    }

private:
    cv::KalmanFilter _kalman;

	//std::deque<cv::Vec4f> _vecs;

	int _mode;
    int _state_num;
    int _measure_num;
};



#endif