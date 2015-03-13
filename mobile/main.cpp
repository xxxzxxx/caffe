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


Net<float> *caffe_net;

template <typename T>
vector<size_t> ordered(vector<T> const& values) {
    vector<size_t> indices(values.size());
    std::iota(begin(indices), end(indices), static_cast<size_t>(0));

    std::sort(
        begin(indices), end(indices),
        [&](size_t a, size_t b) { return values[a] > values[b]; }
    );
    return indices;
}

int init_net(string model_path, string weights_path) {
	Caffe::set_mode(Caffe::CPU);

	clock_t t_start = clock();
	caffe_net = new Net<float>(model_path, caffe::TEST);
	caffe_net->CopyTrainedLayersFrom(weights_path);
	clock_t t_end = clock();
	LOG(DEBUG) << "Loading time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	return 0;
}

vector<int> predict_top_k(string img_path, int k=3) {
    CHECK(caffe_net != NULL);

    Datum datum;
    CHECK(ReadImageToDatum(img_path, 0, 256, 256, true, &datum));
    const shared_ptr<MemoryDataLayer<float>> memory_data_layer =
        static_pointer_cast<MemoryDataLayer<float>>(
            caffe_net->layer_by_name("data"));
    memory_data_layer->AddDatumVector(vector<Datum>({datum}));

    float loss;
    vector<Blob<float>* > dummy_bottom_vec;
    clock_t t_start = clock();
    const vector<Blob<float>*>& result = caffe_net->Forward(dummy_bottom_vec, &loss);
    clock_t t_end = clock();
    LOG(DEBUG) << "Prediction time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

    const vector<float> probs = vector<float>(result[1]->cpu_data(), result[1]->cpu_data() + result[1]->count());
    CHECK_LE(k, probs.size());
    vector<size_t> sorted_index = ordered(probs);

    return vector<int>(sorted_index.begin(), sorted_index.begin() + k);
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
	int top_1 = predict_top_k(string(argv[3]))[0];
	std::cout << "top-1: " << top_1 << std::endl;

	return 0;
}
