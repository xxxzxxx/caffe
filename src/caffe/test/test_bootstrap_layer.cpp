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
class BootstrapLayerTest : public ::testing::Test {
 protected:
  BootstrapLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_p_bootstrap_(new Blob<Dtype>()) {
    Caffe::set_random_seed(4869);
    // fill the values
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_p_bootstrap_);
  }
  virtual ~BootstrapLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_p_bootstrap_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_p_bootstrap_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BootstrapLayerTest, TestDtypes);

TYPED_TEST(BootstrapLayerTest, TestForwardCPUArgMax) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  BootstrapParameter* bootstrap_param =
      layer_param.mutable_bootstrap_param();
  bootstrap_param->set_is_hard_mode(true);
  bootstrap_param->set_beta(0);
  BootstrapLayer<TypeParam> bootstrap_layer(layer_param);
  bootstrap_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  bootstrap_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<TypeParam>* const argmax_top_label_ = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> argmax_bottom_vec_;
  vector<Blob<TypeParam>*> argmax_top_vec_;
  argmax_bottom_vec_.push_back(this->blob_bottom_data_);
  argmax_top_vec_.push_back(argmax_top_label_);
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(1);
  ArgMaxLayer<TypeParam> argmax_layer(layer_param);
  argmax_layer.SetUp(argmax_bottom_vec_, argmax_top_vec_);
  argmax_layer.Forward(argmax_bottom_vec_, argmax_top_vec_);

  const int num = this->blob_top_p_bootstrap_->num();
  const int dim = this->blob_top_p_bootstrap_->count() / num;
  const TypeParam* label_data = argmax_top_label_->cpu_data();
  const TypeParam* top_data = this->blob_top_p_bootstrap_->cpu_data();

  for (int i = 0; i < num; i++) {
    const int label = static_cast<int>(label_data[i]);
    for (int j = 0; j < dim; j++) {
      if (j != label) {
        EXPECT_EQ(top_data[i * dim + j], 0);
      } else {
        EXPECT_EQ(top_data[i * dim + j], 1);
      }
    }
  }
}

TYPED_TEST(BootstrapLayerTest, TestForwardCPUSoftmax) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  BootstrapParameter* bootstrap_param =
      layer_param.mutable_bootstrap_param();
  bootstrap_param->set_is_hard_mode(false);
  bootstrap_param->set_beta(0);
  BootstrapLayer<TypeParam> bootstrap_layer(layer_param);
  bootstrap_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  bootstrap_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<TypeParam>* const softmax_top_data_ = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> softmax_bottom_vec_;
  vector<Blob<TypeParam>*> softmax_top_vec_;
  softmax_bottom_vec_.push_back(this->blob_bottom_data_);
  softmax_top_vec_.push_back(softmax_top_data_);
  SoftmaxLayer<TypeParam> softmax_layer(layer_param);
  softmax_layer.SetUp(softmax_bottom_vec_, softmax_top_vec_);
  softmax_layer.Forward(softmax_bottom_vec_, softmax_top_vec_);

  const int count = this->blob_top_p_bootstrap_->count();
  EXPECT_EQ(softmax_top_data_->count(), count);
  EXPECT_EQ(softmax_top_data_->num(), this->blob_top_p_bootstrap_->num());
  const TypeParam* bootstrap_top_data = this->blob_top_p_bootstrap_->cpu_data();
  const TypeParam* softmax_top_data = softmax_top_data_->cpu_data();

  for (int i = 0; i < count; i++) {
    EXPECT_EQ(softmax_top_data[i], bootstrap_top_data[i]);
  }
}

TYPED_TEST(BootstrapLayerTest, TestForwardCPUGroundtruthLabel) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  BootstrapParameter* bootstrap_param =
      layer_param.mutable_bootstrap_param();
  bootstrap_param->set_is_hard_mode(true);
  bootstrap_param->set_beta(1.0);
  BootstrapLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const int num = this->blob_top_p_bootstrap_->num();
  const int dim = this->blob_top_p_bootstrap_->count() / num;
  const TypeParam* label_data = this->blob_bottom_label_->cpu_data();
  const TypeParam* top_data = this->blob_top_p_bootstrap_->cpu_data();

  for (int i = 0; i < num; i++) {
    const int label = static_cast<int>(label_data[i]);
    for (int j = 0; j < dim; j++) {
      if (j != label) {
        EXPECT_EQ(top_data[i * dim + j], 0);
      } else {
        EXPECT_EQ(top_data[i * dim + j], 1);
      }
    }
  }
}

TYPED_TEST(BootstrapLayerTest, TestHardMode) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  BootstrapParameter* bootstrap_param =
      layer_param.mutable_bootstrap_param();
  bootstrap_param->set_is_hard_mode(true);
  bootstrap_param->set_beta(0.8);
  BootstrapLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<TypeParam>* const argmax_top_label_ = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> argmax_bottom_vec_;
  vector<Blob<TypeParam>*> argmax_top_vec_;
  argmax_bottom_vec_.push_back(this->blob_bottom_data_);
  argmax_top_vec_.push_back(argmax_top_label_);
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(1);
  ArgMaxLayer<TypeParam> argmax_layer(layer_param);
  argmax_layer.SetUp(argmax_bottom_vec_, argmax_top_vec_);
  argmax_layer.Forward(argmax_bottom_vec_, argmax_top_vec_);

  const TypeParam beta = layer_param.bootstrap_param().beta();
  const int num = this->blob_top_p_bootstrap_->num();
  const int dim = this->blob_top_p_bootstrap_->count() / num;
  const TypeParam* n_label_data = this->blob_bottom_label_->cpu_data();
  const TypeParam* p_label_data = argmax_top_label_->cpu_data();
  const TypeParam* top_data = this->blob_top_p_bootstrap_->cpu_data();

  for (int i = 0; i < num; i++) {
    const int n_label = static_cast<int>(n_label_data[i]);
    const int p_label = static_cast<int>(p_label_data[i]);
    for (int j = 0; j < dim; j++) {
      TypeParam p = beta * (j == n_label) + (1-beta) * (j == p_label);
      EXPECT_EQ(top_data[i * dim + j], p);
    }
  }
}

TYPED_TEST(BootstrapLayerTest, TestSoftMode) {
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  BootstrapParameter* bootstrap_param =
      layer_param.mutable_bootstrap_param();
  bootstrap_param->set_is_hard_mode(false);
  bootstrap_param->set_beta(0.8);
  BootstrapLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<TypeParam>* const softmax_top_data_ = new Blob<TypeParam>();
  vector<Blob<TypeParam>*> softmax_bottom_vec_;
  vector<Blob<TypeParam>*> softmax_top_vec_;
  softmax_bottom_vec_.push_back(this->blob_bottom_data_);
  softmax_top_vec_.push_back(softmax_top_data_);
  SoftmaxLayer<TypeParam> softmax_layer(layer_param);
  softmax_layer.SetUp(softmax_bottom_vec_, softmax_top_vec_);
  softmax_layer.Forward(softmax_bottom_vec_, softmax_top_vec_);

  const TypeParam beta = layer_param.bootstrap_param().beta();
  const int num = this->blob_top_p_bootstrap_->num();
  const int dim = this->blob_top_p_bootstrap_->count() / num;
  const TypeParam* n_label_data = this->blob_bottom_label_->cpu_data();
  const TypeParam* prob_data = softmax_top_data_->cpu_data();
  const TypeParam* top_data = this->blob_top_p_bootstrap_->cpu_data();

  for (int i = 0; i < num; i++) {
    const int n_label = static_cast<int>(n_label_data[i]);
    for (int j = 0; j < dim; j++) {
      TypeParam p = beta * (j == n_label) + (1-beta) * prob_data[i * dim + j];
      EXPECT_NEAR(top_data[i * dim + j], p, 1e-6 * top_data[i * dim + j]);
    }
  }
}

}  // namespace caffe
