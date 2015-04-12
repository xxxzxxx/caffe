#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BootstrappingLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  LayerParameter bootstrap_param(this->layer_param_);
  bootstrap_param.set_type("Bootstrap");
  bootstrap_layer_ = LayerRegistry<Dtype>::CreateLayer(bootstrap_param);
  bootstrap_bottom_vec_.clear();
  bootstrap_bottom_vec_.push_back(bottom[0]);
  bootstrap_bottom_vec_.push_back(bottom[1]);
  bootstrap_top_vec_.clear();
  bootstrap_top_vec_.push_back(&b_prob_);
  bootstrap_top_vec_.push_back(&prob_);
  bootstrap_layer_->SetUp(bootstrap_bottom_vec_, bootstrap_top_vec_);
}

template <typename Dtype>
void BootstrappingLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if prediction shape is (N, C), "
      << "label count (number of labels) must be N, "
      << "with integer values in {0, 1, ..., C-1}.";
  bootstrap_layer_->Reshape(bootstrap_bottom_vec_, bootstrap_top_vec_);
}

template <typename Dtype>
void BootstrappingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the bootstrap prob values.
  bootstrap_layer_->Forward(bootstrap_bottom_vec_, bootstrap_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const Dtype* p = b_prob_.cpu_data();
  const Dtype* q = prob_.cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
	/*
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    */
    loss -= p[i] * log(std::max(q[i], Dtype(FLT_MIN)));
    loss -= (1 - p[i]) * log(std::max(1 - q[i], Dtype(FLT_MIN)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void BootstrappingLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* p = b_prob_.cpu_data();
    const Dtype* q = prob_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, q, p, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(BootstrappingLossLayer);
REGISTER_LAYER_CLASS(BootstrappingLoss);

}  // namespace caffe
