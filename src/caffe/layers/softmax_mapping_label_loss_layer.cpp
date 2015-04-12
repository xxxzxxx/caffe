#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithMappingLabelLossLayer<Dtype>::LayerSetUp(
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

  N_ = bottom[0]->count() / bottom[0]->num();
  K_ = N_;
  // Check if we need to set up the stochastic matrix
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Intialize the stochastic matrix
    this->blobs_.resize(1);
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the matrix
    shared_ptr<Filler<Dtype> > matrix_filler(GetFiller<Dtype>(
        this->layer_param_.softmax_mapping_label_loss_param().matrix_filler()));
    matrix_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SoftmaxWithMappingLabelLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  // Figure out the dimensions
  M_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with inner product parameters.";
  mapped_prob_.Reshape(M_, N_, 1, 1);
  if (top.size() >= 2) top[1]->ReshapeLike(prob_);
  if (top.size() >= 3) top[2]->ReshapeLike(mapped_prob_);
}

template <typename Dtype>
void SoftmaxWithMappingLabelLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  ProjectToStochasticMatrixSpace(this->blobs_[0].get());
  const Dtype* matrix = this->blobs_[0]->cpu_data();
  // Map the probability linearly
  Dtype* mapped_prob_data = mapped_prob_.mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      prob_data, matrix, (Dtype)0., mapped_prob_data);
  // Compute the loss
  Dtype loss = 0;
  for (int i = 0; i < M_; ++i) {
    int l = static_cast<int>(label[i]);
    loss -= log(std::max(mapped_prob_data[i * N_ + l],
                         Dtype(FLT_MIN)));
  }
  top[0]->mutable_cpu_data()[0] = loss / M_;
  if (top.size() >= 2) top[1]->ShareData(prob_);
  if (top.size() >= 3) top[2]->ShareData(mapped_prob_);
}

template <typename Dtype>
void SoftmaxWithMappingLabelLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (this->param_propagate_down_[0]) {
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* mapped_prob_data = mapped_prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* matrix_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_memset(sizeof(Dtype) * this->blobs_[0]->count(), 0, matrix_diff);
    for (int i = 0; i < M_; ++i) {
      int l = static_cast<int>(label[i]);
      for (int j = 0; j < K_; ++j) {
        matrix_diff[l * K_ + j] -= (prob_data[i * K_ + j] /
                                    mapped_prob_data[i * N_ + l]);
      }
    }
    caffe_scal(this->blobs_[0]->count(), Dtype(1.0) / M_, matrix_diff);
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* mapped_prob_data = mapped_prob_.cpu_data();
    const Dtype* matrix = this->blobs_[0]->cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    for (int i = 0; i < M_; ++i) {
      int l = static_cast<int>(label[i]);
      for (int j = 0; j < K_; ++j) {
        bottom_diff[i * K_ + j] -= (matrix[l * K_ + j] *
				                    prob_data[i * K_ + j] /
			                        mapped_prob_data[i * N_ + l]);
      }
    }
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	caffe_scal(prob_.count(), loss_weight / M_, bottom_diff);
  }
}

template <typename Dtype>
void SoftmaxWithMappingLabelLossLayer<Dtype>::ProjectToStochasticMatrixSpace(
    Blob<Dtype>* matrix) {
  Dtype* matrix_data = matrix->mutable_cpu_data();
  for (int j = 0; j < K_; ++j) {
    Dtype col_sum = 0;
    for (int i = 0; i < N_; ++i) {
      col_sum += matrix_data[i * K_ + j];
    }
    for (int i = 0; i < N_; ++i) {
      matrix_data[i * K_ + j] /= col_sum;
    }
  }
}

INSTANTIATE_CLASS(SoftmaxWithMappingLabelLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithMappingLabelLoss);

}  // namespace caffe
