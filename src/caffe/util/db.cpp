#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"

#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
#endif
#ifdef USE_LMDB
  case DataParameter_DB_LMDB:
    return new LMDB();
#endif
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

DB* GetDB(const string& backend) {
#ifdef USE_LEVELDB
  if (backend == "leveldb") {
    return new LevelDB();
  }
#endif
#ifdef USE_LMDB
if (backend == "lmdb") {
    return new LMDB();
  }
#endif
  LOG(FATAL) << "Unknown database backend";
  }

}  // namespace db
}  // namespace caffe
