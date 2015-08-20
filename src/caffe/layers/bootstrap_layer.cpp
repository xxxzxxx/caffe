#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BootstrapLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  LayerParameter layer_param;
  layer_param.set_type("ArgMax");
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(1);
  argmax_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  argmax_top_vec_.clear();
  argmax_top_vec_.push_back(&p_label_);
  argmax_layer_->SetUp(softmax_top_vec_, argmax_top_vec_);

  is_hard_mode_ = this->layer_param_.bootstrap_param().mode() == BootstrapParameter_Mode_HARD;
  beta_ = this->layer_param_.bootstrap_param().beta();
}

template <typename Dtype>
void BootstrapLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  argmax_layer_->Reshape(softmax_top_vec_, argmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BootstrapLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  argmax_layer_->Forward(softmax_top_vec_, argmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* p_label = p_label_.cpu_data();  // predicted label
  const Dtype* n_label = bottom[1]->cpu_data();  // noisy label
  const int batch_size = bottom[0]->num();
  const int dim = prob_.count() / batch_size;

  Dtype* p_bootstrap = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), p_bootstrap);

  if (is_hard_mode_) {
    for (int i = 0; i < batch_size; ++i) {
      const int n_label_value = static_cast<int>(n_label[i]);
      const int p_label_value = static_cast<int>(p_label[i]);
      if (n_label_value != p_label_value) {
        p_bootstrap[i * dim + p_label_value] = (1 - beta_);
        p_bootstrap[i * dim + n_label_value] = beta_;
      } else {
        p_bootstrap[i * dim + n_label_value] = 1;
      }
    }
  } else {
    for (int i = 0; i < batch_size; ++i) {
      const int n_label_value = static_cast<int>(n_label[i]);
      p_bootstrap[i * dim + n_label_value] = 1;
    }
    caffe_cpu_axpby(top[0]->count(), 1-beta_, prob_data, beta_, p_bootstrap);
  }
}

template <typename Dtype>
void BootstrapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to prob inputs.";
  }
}

INSTANTIATE_CLASS(BootstrapLayer);
REGISTER_LAYER_CLASS(Bootstrap);

}  // namespace caffe
