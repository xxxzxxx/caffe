#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/glog_alternate.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::shared_ptr;
using caffe::vector;
using caffe::MemoryDataLayer;

using std::string;
using std::static_pointer_cast;
using std::clock;
using std::clock_t;

int test(string model_path, string weights_path, int iter=1) {
	LOG(INFO) << "Use CPU.";
	Caffe::set_mode(Caffe::CPU);

	// Instantiate the caffe net.
	Caffe::set_phase(Caffe::TEST);
	Net<float> caffe_net(model_path);
	caffe_net.CopyTrainedLayersFrom(weights_path);

	LOG(INFO) << "start testing...";
	vector<Blob<float>* > bottom_vec;
	vector<int> test_score_output_id;
	vector<float> test_score;
	float loss = 0;
	for (int i = 0; i < iter; ++i) {
		float iter_loss;
		const vector<Blob<float>*>& result =
		caffe_net.Forward(bottom_vec, &iter_loss);
		loss += iter_loss;
		int idx = 0;
		for (int j = 0; j < result.size(); ++j) {
			const float* result_vec = result[j]->cpu_data();
			for (int k = 0; k < result[j]->count(); ++k, ++idx) {
				const float score = result_vec[k];
				if (i == 0) {
					test_score.push_back(score);
					test_score_output_id.push_back(j);
				} else {
					test_score[idx] += score;
				}
				const std::string& output_name = caffe_net.blob_names()[
				caffe_net.output_blob_indices()[j]];
				LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
			}
		}
	}
	loss /= iter;
	LOG(INFO) << "Loss: " << loss;
	for (int i = 0; i < test_score.size(); ++i) {
		const std::string& output_name = caffe_net.blob_names()[
		caffe_net.output_blob_indices()[test_score_output_id[i]]];
		const float loss_weight =
		caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[i]];
		std::ostringstream loss_msg_stream;
		const float mean_score = test_score[i] / iter;
		if (loss_weight) {
			loss_msg_stream << " (* " << loss_weight
			                << " = " << loss_weight * mean_score << " loss)";
		}
		LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
	}

	return 0;
}

Net<float> *caffe_net;

int init_net(string model_path, string weights_path) {
	Caffe::set_mode(Caffe::CPU);
	Caffe::set_phase(Caffe::TEST);

	clock_t t_start = clock();
	caffe_net = new Net<float>(model_path);
	caffe_net->CopyTrainedLayersFrom(weights_path);
	clock_t t_end = clock();
	LOG(DEBUG) << "Loading time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	return 0;
}

int predict(string img_path, int label=0) {
	CHECK(caffe_net != NULL);

	Datum datum;
	CHECK(ReadImageToDatum(img_path, label, 256, 256, true, &datum));
	const shared_ptr<MemoryDataLayer<float>> memory_data_layer =
		static_pointer_cast<MemoryDataLayer<float>>(
			caffe_net->layer_by_name("data"));
	memory_data_layer->AddDatumVector(vector<Datum>({datum}));

	vector<Blob<float>* > dummy_bottom_vec;
	float loss;
	clock_t t_start = clock();
	const vector<Blob<float>*>& result = caffe_net->Forward(dummy_bottom_vec, &loss);
	clock_t t_end = clock();
	LOG(DEBUG) << "Prediction time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	const float* argmaxs = result[1]->cpu_data();
	for (int i = 0; i < result[1]->num(); i++) {
		for (int j = 0; j < result[1]->height(); j++) {
			LOG(INFO) << " Image: "<< i << " class:"
			          << argmaxs[i*result[1]->height() + j];
		}
	}

	return argmaxs[0];
}

int main(int argc, char const *argv[])
{
	string usage("usage: main <model> <weights> <img>");
	if (argc < 4) {
		std::cerr << usage << std::endl;
		return 1;
	}

	caffe::LogMessage::Enable(true); // enable logging
	// int result = test(string(argv[1]), string(argv[2]), string(argv[3]));
	init_net(string(argv[1]), string(argv[2]));
	predict(string(argv[3]));

	return 0;
}
