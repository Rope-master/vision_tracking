//
// Created by grayson on 18-1-3.
//
#ifndef VISION_DETECTION_SDD_DETECTION_H
#define VISION_DETECTION_SDD_DETECTION_H

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//int a;
namespace ssd_detection
{
    using namespace caffe;  // NOLINT(build/namespaces)

    class SSDDetector
    {
    public:
        SSDDetector() = default;

        SSDDetector(const string &model_file,
                    const string &weights_file,
                    const string &mean_file = "",
                    const string &mean_value = "104,117,123");

        Blob<float> *detect(const cv::Mat &img);

        bool isInitialized();

    private:
        void setMean(const string &mean_file, const string &mean_value);

        void wrapInputLayer(std::vector<cv::Mat> *input_channels);

        void preprocess(const cv::Mat &img,
                        std::vector<cv::Mat> *input_channels);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        bool initialized_ = false;
    };

    SSDDetector::SSDDetector(const string &model_file,
                             const string &weights_file,
                             const string &mean_file,
                             const string &mean_value)
    {
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        Caffe::set_mode(Caffe::GPU);
#endif

        /* Load the network. */
        // TEST means test phase
        net_.reset(new Net<float>(model_file, TEST));
        net_->CopyTrainedLayersFrom(weights_file);

        CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
        CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

        Blob<float> *input_layer = net_->input_blobs()[0];
        num_channels_ = input_layer->channels();
        CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

        /* Load the binaryproto mean file. */
        setMean(mean_file, mean_value);
        initialized_ = true;
    }

    Blob<float> *SSDDetector::detect(const cv::Mat &img)
    {
        Blob<float> *input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,
                             input_geometry_.height, input_geometry_.width);
        /* Forward dimension change to all layers. */
        net_->Reshape();

        std::vector<cv::Mat> input_channels;
        wrapInputLayer(&input_channels);

        preprocess(img, &input_channels);

        net_->Forward();

        /* Copy the output layer to a std::vector */
        Blob<float> *result_blob = net_->output_blobs()[0];
        return result_blob;
    }

    /* Load the mean file in binaryproto format. */
    void SSDDetector::setMean(const string &mean_file, const string &mean_value)
    {
        cv::Scalar channel_mean;
        if (!mean_file.empty())
        {
            CHECK(mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
            BlobProto blob_proto;
            ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

            /* Convert from BlobProto to Blob<float> */
            Blob<float> mean_blob;
            mean_blob.FromProto(blob_proto);
            CHECK_EQ(mean_blob.channels(), num_channels_)
                << "Number of channels of mean file doesn't match input layer.";

            /* The format of the mean file is planar 32-bit float BGR or grayscale. */
            std::vector<cv::Mat> channels;
            float *data = mean_blob.mutable_cpu_data();
            for (int i = 0; i < num_channels_; ++i)
            {
                /* Extract an individual channel. */
                cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
                channels.push_back(channel);
                data += mean_blob.height() * mean_blob.width();
            }

            /* Merge the separate channels into a single image. */
            cv::Mat mean;
            cv::merge(channels, mean);

            /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
            channel_mean = cv::mean(mean);
            mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
        }
        if (!mean_value.empty())
        {
            CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
            stringstream ss(mean_value);
            vector<float> values;
            string item;
            while (getline(ss, item, ','))
            {
                float value = std::atof(item.c_str());
                values.push_back(value);
            }
            CHECK(values.size() == 1 || values.size() == num_channels_) << "Specify either 1 mean_value or as many as channels: "
                                                                        << num_channels_;

            std::vector<cv::Mat> channels;
            for (int i = 0; i < num_channels_; ++i)
            {
                /* Extract an individual channel. */
                cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                                cv::Scalar(values[i]));
                channels.push_back(channel);
            }
            cv::merge(channels, mean_);
        }
    }

    /* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
    void SSDDetector::wrapInputLayer(std::vector<cv::Mat> *input_channels)
    {
        Blob<float> *input_layer = net_->input_blobs()[0];

        int width = input_layer->width();
        int height = input_layer->height();
        float *input_data = input_layer->mutable_cpu_data();
        for (int i = 0; i < input_layer->channels(); ++i)
        {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
    }

    void SSDDetector::preprocess(const cv::Mat &img,
                                 std::vector<cv::Mat> *input_channels)
    {
        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
        {
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        }
        else if (img.channels() == 4 && num_channels_ == 1)
        {
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
        }
        else if (img.channels() == 4 && num_channels_ == 3)
        {
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
        }
        else if (img.channels() == 1 && num_channels_ == 3)
        {
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
        }
        else
        {
            sample = img;
        }

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
        {
            cv::resize(sample, sample_resized, input_geometry_);
        }
        else
        {
            sample_resized = sample;
        }

        cv::Mat sample_float;

        if (num_channels_ == 3)
        {
            sample_resized.convertTo(sample_float, CV_32FC3);
        }
        else
        {
            sample_resized.convertTo(sample_float, CV_32FC1);
        }
        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);

        /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
        cv::split(sample_normalized, *input_channels);

        CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
    }

    bool SSDDetector::isInitialized()
    {
        return initialized_;
    }
}

#endif //VISION_DETECTION_SDD_DETECTION_H
