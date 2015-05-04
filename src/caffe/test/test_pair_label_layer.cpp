#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
class PairLabelLayerTest : public ::testing::Test {
 protected:
  PairLabelLayerTest()
      : blob_bottom_data_l_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_data_r_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_sim_(new Blob<Dtype>()) {
    Caffe::set_random_seed(4869);
    // fill the values
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_l_);
    filler.Fill(this->blob_bottom_data_r_);
    blob_bottom_vec_.push_back(blob_bottom_data_l_);
    blob_bottom_vec_.push_back(blob_bottom_data_r_);
    blob_top_vec_.push_back(blob_top_sim_);
  }
  virtual ~PairLabelLayerTest() {
    delete blob_bottom_data_l_;
    delete blob_bottom_data_r_;
    delete blob_top_sim_;
  }
  Blob<Dtype>* const blob_bottom_data_l_;
  Blob<Dtype>* const blob_bottom_data_r_;
  Blob<Dtype>* const blob_top_sim_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PairLabelLayerTest, TestDtypes);

TYPED_TEST(PairLabelLayerTest, TestZeroThreshold) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  PairLabelParameter* pair_label_param =
      layer_param.mutable_pair_label_param();
  pair_label_param->set_thresh(0);
  PairLabelLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const TypeParam thresh = layer_param.pair_label_param().thresh();
  const int num = this->blob_top_sim_->num();
  const TypeParam* top_data = this->blob_top_sim_->cpu_data();

  for (int i = 0; i < num; i++) {
    EXPECT_NEAR(TypeParam(1), top_data[i], 1e-6);
  }
}

TYPED_TEST(PairLabelLayerTest, TestEqualBottoms) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  PairLabelParameter* pair_label_param =
      layer_param.mutable_pair_label_param();
  pair_label_param->set_thresh(1);
  PairLabelLayer<TypeParam> layer(layer_param);
  this->blob_bottom_vec_[1] = this->blob_bottom_vec_[0];
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const TypeParam thresh = layer_param.pair_label_param().thresh();
  const int num = this->blob_top_sim_->num();
  const TypeParam* top_data = this->blob_top_sim_->cpu_data();

  for (int i = 0; i < num; i++) {
    EXPECT_NEAR(TypeParam(1), top_data[i], 1e-6);
  }
}

TYPED_TEST(PairLabelLayerTest, TestOrthogonal) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  PairLabelParameter* pair_label_param =
      layer_param.mutable_pair_label_param();
  pair_label_param->set_thresh(1e-12);
  PairLabelLayer<TypeParam> layer(layer_param);

  const int dim = this->blob_bottom_data_l_->channels();
  std::vector<int> indices(dim, 0);
  for (int i = 0; i < dim; ++i) {
    indices[i] = i;
  }
  TypeParam* bottom_data_l = this->blob_bottom_data_l_->mutable_cpu_data();
  TypeParam* bottom_data_r = this->blob_bottom_data_r_->mutable_cpu_data();
  caffe_set(this->blob_bottom_data_l_->count(), TypeParam(0), bottom_data_l);
  caffe_set(this->blob_bottom_data_r_->count(), TypeParam(0), bottom_data_r);
  for (int i = 0; i < this->blob_bottom_data_l_->num(); ++i) {
    std::random_shuffle(indices.begin(), indices.end());
    bottom_data_l[i * dim + indices[0]] = TypeParam(1);
    bottom_data_r[i * dim + indices[1]] = TypeParam(1);
  }
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const TypeParam thresh = layer_param.pair_label_param().thresh();
  const int num = this->blob_top_sim_->num();
  const TypeParam* top_data = this->blob_top_sim_->cpu_data();

  for (int i = 0; i < num; i++) {
    EXPECT_EQ(TypeParam(0), top_data[i]);
  }
}

}  // namespace caffe
