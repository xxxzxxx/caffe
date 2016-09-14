#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include <string>
#include <vector>
#include <memory>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>

using std::string;
using std::vector;

namespace caffe {

class CaffeMobile {
public:
  typedef size_t identity_t;

public:
  CaffeMobile() = default;
  virtual ~CaffeMobile() = default;

public:
  static identity_t PutStoreInstance(std::shared_ptr<CaffeMobile> &target);
  static std::shared_ptr<CaffeMobile> FindStoredInstance(const identity_t &identity);
  static void EraseStoredInstance(const identity_t &target);
  static void EraseStoredInstance(const std::shared_ptr<CaffeMobile> &target);

public:
  void SetMean(const string &mean_file);
  void SetMean(const vector<float> &mean_values);
  void SetScale(const float scale);
  vector<float> GetConfidenceScore(const cv::Mat &img);
  vector<int> PredictTopK(const cv::Mat &img, int k);
  vector<vector<float>> ExtractFeatures(const cv::Mat &img,const string &str_blob_names);
  void LoadModule(const string &model_path, const string &weights_path);

private:
  void Preprocess(const cv::Mat &img, vector<cv::Mat> *input_channels);
  void WrapInputLayer(std::vector<cv::Mat> *input_channels);
  vector<float> Forward(const cv::Mat &img);

private:
  shared_ptr<Net<float>> net;
  cv::Size input_geometry;
  int num_channels;
  cv::Mat mean;
  float scale;
};

} // namespace caffe

#endif
