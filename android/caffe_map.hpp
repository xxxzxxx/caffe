#pragma once

#include <map>
#include <vector>
#include "caffe_concurrent.hpp"

namespace caffe
{
	template<typename Key, typename Value>
	class ImmutableMap
	{
	public:
		ImmutableMap() = default;
		ImmutableMap(const ImmutableMap&) = default;
		virtual ~ImmutableMap() = default;

	public:
		virtual bool find(const Key & key, Value& popped_value) const = 0;
		virtual Value find(const Key & key) const = 0;
		virtual bool contains(const Key& key) const = 0;
		virtual std::vector<Value> values() const = 0;
		virtual std::vector<Key> keys() const = 0;
		virtual bool empty() const = 0;
		virtual size_t size() const = 0;
		virtual std::vector<std::pair<Key, Value>> all() const = 0;
	};

	template<typename Key, typename Value> 
	class MutableMap : public ImmutableMap<Key, Value>
	{
	public:
		MutableMap() = default;
		MutableMap(const MutableMap&) = default;
		virtual ~MutableMap() = default;
	public:
		virtual void clear() = 0;
		virtual bool erase(const Key& key) = 0;
		virtual void insert(const Key& key, Value const& data) = 0;
	};

	template<typename Key, typename Value> 
	class MutableMapObject : public MutableMap<Key, Value>
	{
	public:
		typedef std::map<Key, Value> Container;
	private:
		Container container;

	public:
		MutableMapObject() = default;
		MutableMapObject(const MutableMapObject&) = default;
		virtual ~MutableMapObject() = default;

	public:
		void clear() override
		{
			container.clear();
		};
		bool contains(const Key& key) const override
		{
			auto itr = container.find(key);
			return (itr != container.end());
		};
		bool erase(const Key& key) override
		{
			size_t n = container.erase(key);
			return (n > 0);
		};
		size_t size() const override
		{
			return container.size();
		};
		void insert(const Key& key, Value const& data) override
		{
			container[key] = data;
		};
		bool empty() const override
		{
			return container.empty();
		};

		std::vector<Value> values() const override
		{
			std::vector<Value> values;
			auto current = container.begin();
			for (; current != container.end(); ++current)
			{
				values.push_back((*current).second);
			}
			return values;
		};

		std::vector<Key> keys() const override
		{
			std::vector<Key> keys;
			auto current = container.begin();
			for (; current != container.end(); ++current)
			{
				keys.push_back((*current).first);
			}
			return keys;
		};

		std::vector<std::pair<Key, Value>> all() const override
		{
			std::vector<std::pair<Key, Value>> result;
			auto current = container.begin();
			for (; current != container.end(); ++current)
			{
				std::pair<Key, Value> b = *current;
				result.push_back(b);
			}
			return result;
		}

		bool find(const Key & key, Value& popped_value) const override
		{
			auto itr = container.find(key);
			bool result = false;
			result = (itr != container.end());
			if (result)
			{
				popped_value = itr->second;
			}
			return result;
		};

		Value find(const Key & key) const override
		{
			Value result;
			auto itr = container.find(key);
			if ((itr != container.end()))
			{
				result = itr->second;
			}
			return result;
		};
	};

	template<typename Key, typename Value> 
	class ConcurrentMutableMapObject : public concurrent::BlockingObject, public MutableMap<Key, Value>
	{
	public:
		typedef std::map<Key, Value> Container;
	private:
		Container container;

	public:
		ConcurrentMutableMapObject() = default;
		ConcurrentMutableMapObject(const ConcurrentMutableMapObject&) = default;
		virtual ~ConcurrentMutableMapObject() = default;

	public:
		void clear() override
		{
			std::lock_guard<std::mutex> lock(mtx);
			container.clear();
		};
		bool contains(const Key& key) const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			auto itr = container.find(key);
			return (itr != container.end());
		};
		bool erase(const Key& key) override
		{
			std::lock_guard<std::mutex> lock(mtx);
			size_t n = container.erase(key);
			return (n > 0);
		};
		size_t size() const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			return container.size();
		};
		void insert(const Key& key, Value const& data) override
		{
			std::unique_lock<std::mutex> lock(mtx);
			container[key] = data;
			lock.unlock();
			condition.notify_one();
		};
		bool empty() const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			return container.empty();
		};

		std::vector<Value> values() const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			std::vector<Value> values;
			auto current = container.begin();
			for (; current != container.end(); ++current)
			{
				values.push_back((*current).second);
			}
			return values;
		};

		std::vector<Key> keys() const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			std::vector<Key> keys;
			auto current = container.begin();
			for (; current != container.end(); ++current)
			{
				keys.push_back((*current).first);
			}
			return keys;
		};

		std::vector<std::pair<Key, Value>> all() const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			std::vector<std::pair<Key, Value>> result;
			auto current = container.begin();
			for (; current != container.end(); ++current)
			{
				std::pair<Key, Value> b = *current;
				result.push_back(b);
			}
			return result;
		}

		bool find(const Key & key, Value& popped_value) const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			auto itr = container.find(key);
			bool result = false;
			result = (itr != container.end());
			if (result)
			{
				popped_value = itr->second;
			}
			return result;
		};

		Value find(const Key & key) const override
		{
			std::lock_guard<std::mutex> lock(mtx);
			Value result;
			auto itr = container.find(key);
			if ((itr != container.end()))
			{
				result = itr->second;
			}
			return result;
		};
	};
};