/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#ifndef UCUDNN_CONV_DATABASE_H_
#define UCUDNN_CONV_DATABASE_H_

#include <map>

#include "util.h"
#include "convParam.h"
#include "optCache.h"

namespace ucudnn {

  enum DatabaseCellType {
    BOOL, INT, DOUBLE, STRING
  };

  class DatabaseCell {
  public:
    DatabaseCell(const bool b)        { b_ = b ? 1 : 0;      type_ = BOOL; }
    DatabaseCell(const int i)         { i_ = i;              type_ = INT; }
    DatabaseCell(const size_t i)        { i_ = i;              type_ = INT; }
    DatabaseCell(const double d)      { d_ = d;              type_ = DOUBLE; }
    DatabaseCell(const std::string s) { s_ = s;              type_ = STRING; }
    DatabaseCell(const char *c)       { s_ = std::string(c); type_ = STRING; }
    DatabaseCell(const std::string data, const DatabaseCellType type) {
      switch(type) {
      case BOOL:
	b_ = (std::stol(data) == 1);
	break;
      case INT:
	i_ = std::stol(data);
	break;
      case DOUBLE:
	d_ = std::stod(data);
	break;
      case STRING:
	s_ = data;
      }
      type_ = type;
    }

    operator bool()        const { assert(type_ == BOOL);   return b_; }
    operator int()         const { assert(type_ == INT);    return i_; }
    operator size_t()      const { assert(type_ == INT);    return i_; }
    operator float()       const { assert(type_ == DOUBLE); return d_; }
    operator double()      const { assert(type_ == DOUBLE); return d_; }
    operator std::string() const { assert(type_ == STRING); return s_; }

    std::string toString() const {
      switch(type_) {
      case BOOL:
	return std::to_string(b_);
      case INT:
	return std::to_string(i_);
      case DOUBLE:
	return std::to_string(d_);
      case STRING:
	return std::string("'") + s_ + "'";
      }
      UCUDNN_ERROR_EXIT("invalid DatabaseCell type.");
    }

#ifdef UCUDNN_USE_SQLITE
    void setToSQLiteStmt(sqlite3_stmt *stmt, const int id) {
      switch(type_) {
      case BOOL:
	UCUDNN_SQLITE_CHECK(sqlite3_bind_int(stmt, id, b_ ? 1 : 0));
      case INT:
	UCUDNN_SQLITE_CHECK(sqlite3_bind_int(stmt, id, i_));
	return;
      case DOUBLE:
	UCUDNN_SQLITE_CHECK(sqlite3_bind_double(stmt, id, d_));
	return;
      case STRING:
	UCUDNN_SQLITE_CHECK(sqlite3_bind_text(stmt, id, s_.c_str(), -1, SQLITE_STATIC));
	return;
      }
      UCUDNN_ERROR_EXIT("invalid DatabaseCell type.");
    }
#endif

    DatabaseCellType type() { return type_; }

  private:
    bool b_;
    long i_;
    double d_;
    std::string s_;
    DatabaseCellType type_;
  };

  class DatabaseEntry {
  public:
    DatabaseEntry() {};

    void setColumn(const std::string column) {
      entry_.insert(std::make_pair(column, 0));
    }
    template <typename T>
    void setColumn(const std::string column, const T data) {
      entry_.insert(std::make_pair(column, data));
    }
    void setColumn(const std::string column, const std::string data, const DatabaseCellType type) {
      entry_.insert(std::make_pair(column, DatabaseCell(data, type)));
    }

    void setColumnAndType(const std::string column, const DatabaseCellType type) {
      setColumn(column, "0", type);
    }


    DatabaseCell getColumn(const std::string column) const {
      return entry_.at(column);
    }

    std::vector<std::string> keys() const {
      std::vector<std::string> k;
      for(const auto p : entry_)
	k.push_back(p.first);
      return k;
    }

    std::string toString() const {
      std::string ret;
      for(const auto key : keys())
	ret += key + "=" + getColumn(key).toString() +", ";
      return ret;
    }

  private:
    std::map<std::string, DatabaseCell> entry_;
  };

  class ConvDatabase {
  public:
    ConvDatabase(const std::string path);
    ConvDatabase(const ConvDatabase &database);
    ~ConvDatabase();

    void createTables();
    void insertPerfResults(const ConvParam convParam, const ConvType convType, const int batchSize,
			   const size_t workspaceSize,
			   const std::vector<cudnnConvolutionGenericAlgoPerf_t> perfs);
    std::vector<cudnnConvolutionGenericAlgoPerf_t>
    selectPerfResults(const ConvParam convParam, const ConvType convType, const int batchSize, const size_t workspaceSize);

    void setLayerParams(const OptCache &optCache);
    // These functions are vulnerable to SQL injections. Please keep them private
    void deleteEntries(const std::string tableName);
    void createTableIfNotExist(const std::string tableName,
			       const std::vector<std::pair<std::string, std::string> > columns);
    void query(const std::string sql);
    void insert(const std::string tableName, const DatabaseEntry entry);
    std::vector<DatabaseEntry> selectEquals(const std::string tableName,
					    const DatabaseEntry columns,
					    const DatabaseEntry condition);

  private:
    void open();

    const std::string path_;
#ifdef UCUDNN_USE_SQLITE
    sqlite3 *db_;
#endif

    static const std::string layerTableName;
    static const std::string perfTableName;

    static const std::string cName_layerId;
    static const std::string cName_convParam;
    static const std::string cName_convType;
    static const std::string cName_batchSize;
    static const std::string cName_workspaceSize;
    static const std::string cName_cudnnVersion;
    static const std::string cName_gpuName;

    static const std::string cName_perfRank;
    static const std::string cName_perfAlgo;
    static const std::string cName_perfTime;
    static const std::string cName_perfMemory;
    static const std::string cName_perfDeterminism;
    static const std::string cName_perfMathType;
  };

}

#endif
