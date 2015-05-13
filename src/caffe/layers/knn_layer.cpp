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
void KNNLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.knn_param().top_k();
}

template <typename Dtype>
void KNNLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> label_shape(1, bottom[0]->num());

  CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
  if (bottom.size() == 1) {
    CHECK_LE(top_k_, bottom[0]->num()-2)
        << "top_k must be less than or equal to (batch_size - 2).";
    CHECK_EQ(top.size(), 4);

    top[0]->ReshapeLike(*bottom[0]);  // neighbors
    top[1]->ReshapeLike(*bottom[0]);  // non-neighbors
    top[2]->Reshape(label_shape);  // 1s, for neighbors
    top[3]->Reshape(label_shape);  // 0s, for non-neighbors
  } else {
    CHECK_LE(top_k_, bottom[0]->num())
        << "top_k must be less than or equal to batch_size.";
    CHECK(bottom[1]->shape() == bottom[0]->shape());
    CHECK_EQ(top.size(), 1);

    top[0]->Reshape(label_shape);
  }

  vector<int> diff_shape(1, bottom[0]->count() / bottom[0]->num());
  diff_.Reshape(diff_shape);
}

template <typename Dtype>
void KNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int dim = bottom[0]->count() / bottom[0]->num();
  const int batch_size = bottom[0]->num();

  if (bottom.size() == 1) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_neighbors = top[0]->mutable_cpu_data();
    Dtype* top_non_neighbors = top[1]->mutable_cpu_data();
    Dtype* top_neighbor_labels = top[2]->mutable_cpu_data();
    Dtype* top_non_neighbor_labels = top[3]->mutable_cpu_data();

    // compute L2-distance matrix
    vector<vector<Dtype> > l2_dist_mat(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      l2_dist_mat[i] = vector<Dtype>(batch_size, Dtype(0));
    }
    for (int i = 0; i < batch_size - 1; ++i) {
      for (int j = i + 1; j < batch_size; ++j) {
        caffe_sub(
            dim,
            &bottom_data[i * dim],
            &bottom_data[j * dim],
            diff_.mutable_cpu_data());
        l2_dist_mat[i][j] =
            sqrt(caffe_cpu_dot(dim, diff_.cpu_data(), diff_.cpu_data()));
        l2_dist_mat[j][i] = l2_dist_mat[i][j];  // symmetric matrix
      }
    }

    int count;
    std::vector<std::pair<Dtype, int> > dist_vector(batch_size-1);
    for (int i = 0; i < batch_size; ++i) {
      count = 0;
      for (int j = 0; j < batch_size; ++j) {
        if (j == i) {
          continue;
        }
        dist_vector[count++] = std::make_pair(l2_dist_mat[i][j], j);
      }

      std::partial_sort(
          dist_vector.begin(), dist_vector.begin() + top_k_,
          dist_vector.end(), std::less<std::pair<Dtype, int> >());

      unsigned int n_idx = dist_vector[caffe_rng_rand() % top_k_].second;
      unsigned int nn_idx = dist_vector[top_k_ +
          caffe_rng_rand() % (batch_size - top_k_ - 1)].second;

      caffe_copy(dim, &bottom_data[n_idx * dim], &top_neighbors[i * dim]);
      caffe_copy(dim, &bottom_data[nn_idx * dim], &top_non_neighbors[i * dim]);

      top_neighbor_labels[i] = Dtype(1);
      top_non_neighbor_labels[i] = Dtype(0);
    }
  } else {
    const Dtype* l_data = bottom[0]->cpu_data();  // labeled data
    const Dtype* u_data = bottom[1]->cpu_data();  // unlabeld data
    Dtype* top_labels = top[0]->mutable_cpu_data();

    std::vector<std::pair<Dtype, int> > dist_vector(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < batch_size; ++j) {
        caffe_sub(
            dim,
            &u_data[i * dim],
            &l_data[j * dim],
            diff_.mutable_cpu_data());
        Dtype d_sq = caffe_cpu_dot(dim, diff_.cpu_data(), diff_.cpu_data());
        dist_vector[j] = std::make_pair(d_sq, j);
      }

      std::partial_sort(
          dist_vector.begin(), dist_vector.begin() + top_k_,
          dist_vector.end(), std::less<std::pair<Dtype, int> >());

      top_labels[i] = Dtype(0);
      for (int j = 0; j < top_k_; ++j) {
        if (dist_vector[j].second == i) {
          top_labels[i] = Dtype(1);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(KNNLayer);
REGISTER_LAYER_CLASS(KNN);

}  // namespace caffe
