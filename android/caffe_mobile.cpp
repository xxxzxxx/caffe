#include <algorithm>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"

#include "caffe_mobile.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe_map.hpp"

using std::clock;
using std::clock_t;
using std::string;
using std::vector;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::MemoryDataLayer;

namespace caffe {

  static ConcurrentMutableMapObject<CaffeMobile::identity_t,std::shared_ptr<CaffeMobile>> global_caffe_mobile_store;

  template <typename T> vector<int> argmax(vector<T> const &values, int N) {
    vector<size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));
    std::partial_sort(indices.begin(), indices.begin() + N, indices.end(),
                      [&](size_t a, size_t b) { return values[a] > values[b]; });
    return vector<int>(indices.begin(), indices.begin() + N);
  }

  CaffeMobile::identity_t CaffeMobile::PutStoreInstance(std::shared_ptr<CaffeMobile> &target)
  {
    if (target == nullptr)
    {
      throw -1;
    }
    identity_t identity = reinterpret_cast<identity_t>(target.get());
    global_caffe_mobile_store.insert(identity, target);
    return identity;
  }

  std::shared_ptr<CaffeMobile> CaffeMobile::FindStoredInstance(const identity_t &identity)
  {
    std::shared_ptr<CaffeMobile> find = nullptr;
    auto found = global_caffe_mobile_store.find(identity, find);
    return find;
  }
  
  void CaffeMobile::EraseStoredInstance(const identity_t &target)
  {
    global_caffe_mobile_store.erase(target);
  }
  
  void CaffeMobile::EraseStoredInstance(const std::shared_ptr<CaffeMobile> &target)
  {
    if (target != nullptr)
    {
      EraseStoredInstance(reinterpret_cast<identity_t>(target.get()));
    }
  }

  void CaffeMobile::LoadModule(const string &model_path, const string &weights_path) {
    CHECK_GT(model_path.size(), 0) << "Need a model definition to score.";
    CHECK_GT(weights_path.size(), 0) << "Need model weights to score.";

    Caffe::set_mode(Caffe::CPU);

    clock_t t_start = clock();
    this->net.reset(new Net<float>(model_path, caffe::TEST));
    this->net->CopyTrainedLayersFrom(weights_path);
    clock_t t_end = clock();
    LOG(INFO) << "Loading time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
              << " ms.";

    CHECK_EQ(this->net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(this->net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = this->net->input_blobs()[0];
    this->num_channels = input_layer->channels();
    CHECK(this->num_channels == 3 || this->num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
    this->input_geometry = cv::Size(input_layer->width(), input_layer->height());

    this->scale = 0.0;
  }

  void CaffeMobile::SetMean(const vector<float> &mean_values) {
    CHECK_EQ(mean_values.size(), this->num_channels)
        << "Number of mean values doesn't match channels of input layer.";

    cv::Scalar channel_mean(0);
    double *ptr = &channel_mean[0];
    for (int i = 0; i < this->num_channels; ++i) {
      ptr[i] = mean_values[i];
    }
    this->mean = cv::Mat(this->input_geometry, (this->num_channels == 3 ? CV_32FC3 : CV_32FC1), channel_mean);
  }

  void CaffeMobile::SetMean(const string &mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), this->num_channels)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < this->num_channels; ++i) {
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
    cv::Scalar channel_mean = cv::mean(mean);
    this->mean = cv::Mat(this->input_geometry, mean.type(), channel_mean);
  }

  void CaffeMobile::SetScale(const float scale) {
    CHECK_GT(scale, 0);
    this->scale = scale;
  }

  void CaffeMobile::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && this->num_channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && this->num_channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && this->num_channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && this->num_channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
      sample = img;

    cv::Mat sample_resized;
    if (sample.size() != this->input_geometry)
      cv::resize(sample, sample_resized, this->input_geometry);
    else
      sample_resized = sample;

    cv::Mat sample_float;
    if (this->num_channels == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    if (!this->mean.empty()) {
      cv::subtract(sample_float, this->mean, sample_normalized);
    } else {
      sample_normalized = sample_float;
    }

    if (this->scale > 0.0) {
      sample_normalized *= this->scale;
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) ==
          this->net->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
  }

  void CaffeMobile::WrapInputLayer(std::vector<cv::Mat> *input_channels) {
    Blob<float> *input_layer = this->net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
  }

  vector<float> CaffeMobile::Forward(const cv::Mat &img) {
    CHECK(!img.empty()) << "img should not be empty";

    Blob<float> *input_layer = this->net->input_blobs()[0];
    input_layer->Reshape(1, this->num_channels, this->input_geometry.height,
                        this->input_geometry.width);
    /* Forward dimension change to all layers. */
    this->net->Reshape();

    vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    clock_t t_start = clock();
    this->net->Forward();
    clock_t t_end = clock();
    LOG(INFO) << "Forwarding time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
              << " ms.";

    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = this->net->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    return vector<float>(begin, end);
  }

  vector<float> CaffeMobile::GetConfidenceScore(const cv::Mat &img) {
    return Forward(img);
  }

  vector<int> CaffeMobile::PredictTopK(const cv::Mat &img, int k) {
    const vector<float> probs = Forward(img);
    k = std::min<int>(std::max(k, 1), probs.size());
    return argmax(probs, k);
  }

  vector<vector<float>>
  CaffeMobile::ExtractFeatures(const cv::Mat &img,
                              const string &str_blob_names) {
    Forward(img);

    vector<std::string> blob_names;
    boost::split(blob_names, str_blob_names, boost::is_any_of(","));

    size_t num_features = blob_names.size();
    for (size_t i = 0; i < num_features; i++) {
      CHECK(this->net->has_blob(blob_names[i])) << "Unknown feature blob name "
                                          << blob_names[i];
    }

    vector<vector<float>> features;
    for (size_t i = 0; i < num_features; i++) {
      const shared_ptr<Blob<float>> &feat = this->net->blob_by_name(blob_names[i]);
      features.push_back(
          vector<float>(feat->cpu_data(), feat->cpu_data() + feat->count()));
    }

    return features;
  }

} // namespace caffe

/*
using caffe::CaffeMobile;
int main(int argc, char const *argv[]) {
  string usage("usage: main <model> <weights> <mean_file> <img>");
  if (argc < 5) {
    std::cerr << usage << std::endl;
    return 1;
  }

  CaffeMobile *caffe_mobile =
      CaffeMobile::Get(string(argv[1]), string(argv[2]));
  caffe_mobile->SetMean(string(argv[3]));
  vector<int> top_3 = caffe_mobile->PredictTopK(cv::imread(string(argv[4]), -1), 3);
  for (auto i : top_3) {
    std::cout << i << std::endl;
  }
  return 0;
}
*/