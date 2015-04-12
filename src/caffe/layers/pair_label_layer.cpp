#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PairLabelLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  thresh_ = this->layer_param_.pair_label_param().thresh();
}

template <typename Dtype>
void PairLabelLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  vector<int> label_shape(1, bottom[0]->num());
  top[0]->Reshape(label_shape);
}

template <typename Dtype>
void PairLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* repr_first = bottom[0]->cpu_data();
  const Dtype* repr_second = bottom[1]->cpu_data();
  const int batch_size = bottom[0]->num();
  const int dim = bottom[0]->count() / batch_size;
  Dtype* pair_label = top[0]->mutable_cpu_data();

  for (int i = 0; i < batch_size; i++) {
    Dtype cos_sim = (
        caffe_cpu_dot(dim, &repr_first[i * dim], &repr_second[i * dim]) /
        sqrt(caffe_cpu_dot(dim, &repr_first[i * dim], &repr_first[i * dim]) *
             caffe_cpu_dot(dim, &repr_second[i * dim], &repr_second[i * dim]))
        );
    pair_label[i] = Dtype(cos_sim >= thresh_);
  }
}

template <typename Dtype>
void PairLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to repr inputs.";
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to repr inputs.";
  }
}

INSTANTIATE_CLASS(PairLabelLayer);
REGISTER_LAYER_CLASS(PairLabel);

}  // namespace caffe
