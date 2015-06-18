#ifndef CAFFE_BOOST_OR_STD_HPP_
#define CAFFE_BOOST_OR_STD_HPP_

#ifdef USE_BOOST

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

// include/caffe/util/benchmark.hpp
#include <boost/date_time/posix_time/posix_time.hpp>

// src/caffe/util/math_functions.cpp
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

// include/caffe/util/rng.hpp
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"

// include/caffe/internal_thread.hpp
/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }
#include <boost/thread.hpp>

#else

#include <memory>

// include/caffe/internal_thread.hpp
#include <thread>

// include/caffe/util/benchmark.hpp
#include <chrono>

// include/caffe/util/rng.hpp
// src/caffe/util/math_functions.cpp
#include <random>

#endif  // USE_BOOST

namespace caffe {

#ifdef USE_BOOST

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

using boost::scoped_ptr;
using boost::thread;

// include/caffe/util/rng.hpp
typedef boost::mt19937 rng_t;

#else

using std::shared_ptr;

template <typename T>
using scoped_ptr = typename std::unique_ptr<T>;

typedef std::thread thread;

// include/caffe/util/rng.hpp
typedef std::mt19937_64 rng_t;

#endif  // USE_BOOST

}  // namespace caffe

#endif  // CAFFE_BOOST_OR_STD_HPP_
