#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
#ifndef CPU_ONLY
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
#ifdef USE_BOOST
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime stop_cpu_;
#else
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds microseconds;
  typedef std::chrono::milliseconds milliseconds;
  clock::time_point start_cpu_;
  clock::time_point stop_cpu_;
#endif    // USE_BOOST
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
