#pragma once

#include <map>
#include <mutex>
#include <condition_variable>

namespace caffe
{
	namespace concurrent
	{
		class BlockingObject
		{
		public:
			BlockingObject() = default;
			BlockingObject(const BlockingObject&) = default;
			virtual ~BlockingObject() = default;
		protected:
			mutable std::mutex mtx;
			std::condition_variable condition;
		};
	};
};