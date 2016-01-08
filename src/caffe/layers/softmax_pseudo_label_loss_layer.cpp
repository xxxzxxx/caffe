#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithPseudoLabelLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_param.clear_loss_weight();
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

  normalize_ = this->layer_param_.loss_param().normalize();

  if (top.size() == 1) {
    real_weight_ = pseudo_weight_ = this->layer_param_.loss_weight(0);
  } else {
    real_weight_ = this->layer_param_.loss_weight(0);
    base_pseudo_weight_ = this->layer_param_.loss_weight(1);
    T1_ = this->layer_param_.softmax_pseudo_label_loss_param().t1();
    T2_ = this->layer_param_.softmax_pseudo_label_loss_param().t2();
    CHECK_LE(T1_, T2_) << "T1 should not be larger than T2.";
    alpha_ = this->layer_param_.softmax_pseudo_label_loss_param().alpha();
    CHECK_GE(alpha_, Dtype(0)) << "alpha should be non-negative.";
    t_ = 0;
  }
}

template <typename Dtype>
void SoftmaxWithPseudoLabelLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  argmax_layer_->Reshape(softmax_top_vec_, argmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->Reshape(std::vector<int>(1, 1));
  }
}

template <typename Dtype>
void SoftmaxWithPseudoLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  argmax_layer_->Forward(softmax_top_vec_, argmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* p_label = p_label_.cpu_data();  // predicted label
  int dim = prob_.count() / outer_num_;

  // deterministic annealing
  if (this->phase_ == TRAIN && top.size() >= 2) {
    t_++;
    pseudo_weight_ = base_pseudo_weight_;
    if (t_ < T1_) {
      pseudo_weight_ = 0;
    } else if (t_ < T2_) {
      pseudo_weight_ = (Dtype(t_ - T1_) / (T2_ - T1_)) * alpha_ *
                       base_pseudo_weight_;
    } else {
      pseudo_weight_ = alpha_ * base_pseudo_weight_;
    }
  }

  int num_pseudo = 0;
  Dtype loss_real = 0, loss_pseudo = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int l = static_cast<int>(label[i * inner_num_ + j]);
      if (l == -1) {
        // pseudo label
        const int pseudo_label = static_cast<int>(p_label[i * inner_num_ + j]);
        loss_pseudo -= log(
            std::max(prob_data[i * dim + pseudo_label * inner_num_ + j],
                     Dtype(FLT_MIN)));
        ++num_pseudo;
      } else {
        // real label
        DCHECK_GE(l, 0);
        DCHECK_LT(l, prob_.shape(softmax_axis_));
        loss_real -= log(std::max(prob_data[i * dim + l * inner_num_ + j],
                                  Dtype(FLT_MIN)));
      }
    }
  }
  if (num_pseudo > 0) {
    loss_pseudo /= num_pseudo;
  }
  if (outer_num_ * inner_num_ - num_pseudo > 0) {
    loss_real /= (outer_num_ * inner_num_ - num_pseudo);
  }
  if (top.size() == 1) {
    top[0]->mutable_cpu_data()[0] = loss_real + loss_pseudo;
  } else {
    top[0]->mutable_cpu_data()[0] = loss_real;
    top[1]->mutable_cpu_data()[0] = loss_pseudo;
  }
}

template <typename Dtype>
void SoftmaxWithPseudoLabelLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* p_label = p_label_.cpu_data();  // predicted label
    int dim = prob_.count() / outer_num_;
    int channels = prob_.channels();
    int num_pseudo = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        int l = static_cast<int>(label[i * inner_num_ + j]);
        if (l == -1) ++num_pseudo;
      }
    }
    Dtype real_weight = real_weight_ / std::max(outer_num_ * inner_num_ - num_pseudo, 1);
    Dtype pseudo_weight = pseudo_weight_ / std::max(num_pseudo, 1);
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        int l = static_cast<int>(label[i * inner_num_ + j]);
        if (l == -1) {
          // pseudo label
          const int pseudo_label = static_cast<int>(p_label[i * inner_num_ + j]);
          bottom_diff[i * dim + pseudo_label * inner_num_ + j] -= 1;
          for (int k = 0; k < channels; ++k) {
            bottom_diff[i * dim + k * inner_num_ + j] *= pseudo_weight;
          }
        } else {
          // real label
          bottom_diff[i * dim + l * inner_num_ + j] -= 1;
          for (int k = 0; k < channels; ++k) {
            bottom_diff[i * dim + k * inner_num_ + j] *= real_weight;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(SoftmaxWithPseudoLabelLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithPseudoLabelLoss);

}  // namespace caffe
