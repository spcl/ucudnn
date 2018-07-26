/*
 * u-cuDNN: A wrapper library for NVIDIA cuDNN library.
 * Copyright (c) 2018 ETH-Zurich and Tokyo Institute of Technology. All rights reserved.
 * See LICENSE for license information.
 */

#include "convDatabase.h"

#ifdef UCUDNN_USE_SQLITE
#include <sqlite3.h>
#endif

namespace ucudnn {

  const std::string ConvDatabase::cName_layerId = "layerId";

  const std::string ConvDatabase::cName_convParam     = "convParam";
  const std::string ConvDatabase::cName_convType      = "convType";
  const std::string ConvDatabase::cName_batchSize     = "batchSize";
  const std::string ConvDatabase::cName_workspaceSize = "workspaceSize";
  const std::string ConvDatabase::cName_cudnnVersion  = "cudnnVersion";
  const std::string ConvDatabase::cName_gpuName       = "gpuName";

  const std::string ConvDatabase::cName_perfRank        = "perfRank";
  const std::string ConvDatabase::cName_perfAlgo        = "perfAlgo";
  const std::string ConvDatabase::cName_perfTime        = "perfTime";
  const std::string ConvDatabase::cName_perfMemory      = "perfMemory";
  const std::string ConvDatabase::cName_perfDeterminism = "perfDeterminism";
#if CUDNN_HAS_MATHTYPE
  const std::string ConvDatabase::cName_perfMathType    = "perfMathType";
#endif

  const std::string ConvDatabase::layerTableName = "layerTable";
  const std::string ConvDatabase::perfTableName = "perfTable";

  ConvDatabase::ConvDatabase(const std::string path) : path_(path) {
    open();
  }

  ConvDatabase::ConvDatabase(const ConvDatabase &database) : path_(database.path_) {
    open();
  }

  ConvDatabase::~ConvDatabase() {
#ifdef UCUDNN_USE_SQLITE
    UCUDNN_SQLITE_CHECK(sqlite3_close_v2(db_));
#endif
  }

  void ConvDatabase::open() {
#ifdef UCUDNN_USE_SQLITE
    UCUDNN_SQLITE_CHECK(sqlite3_open(path_.c_str(), &db_));
    createTables();
#else
    UCUDNN_ERROR_EXIT("Failed to run instantiate ConvDatabase. Please reinstall with CUDNN_USE_SQLITE option.");
#endif
  }

  void ConvDatabase::createTables() {
#ifdef UCUDNN_USE_SQLITE
    const std::string sqlIntegerType = "INTEGER";
    const std::string sqlBoolType    = "INTEGER";
    const std::string sqlFloatType   = "REAL";
    const std::string sqlStringType  = "TEXT";

    {
      std::vector<std::pair<std::string, std::string> > columns;

      columns.push_back(std::make_pair(cName_convParam,     sqlStringType));
      columns.push_back(std::make_pair(cName_convType,      sqlStringType));
      columns.push_back(std::make_pair(cName_batchSize,     sqlIntegerType));
      columns.push_back(std::make_pair(cName_workspaceSize, sqlIntegerType));
      columns.push_back(std::make_pair(cName_cudnnVersion,  sqlIntegerType));
      columns.push_back(std::make_pair(cName_gpuName,       sqlStringType));

      columns.push_back(std::make_pair(cName_perfRank,        sqlIntegerType));
      columns.push_back(std::make_pair(cName_perfAlgo,        sqlIntegerType));
      columns.push_back(std::make_pair(cName_perfTime,        sqlFloatType));
      columns.push_back(std::make_pair(cName_perfMemory,      sqlIntegerType));
      columns.push_back(std::make_pair(cName_perfDeterminism, sqlBoolType));
#if CUDNN_HAS_MATHTYPE
      columns.push_back(std::make_pair(cName_perfMathType,    sqlBoolType));
#endif

      createTableIfNotExist(perfTableName, columns);
    }

    {
      std::vector<std::pair<std::string, std::string> > columns;
      columns.push_back(std::make_pair(cName_layerId,   sqlIntegerType));
      columns.push_back(std::make_pair(cName_convParam, sqlStringType));
      columns.push_back(std::make_pair(cName_convType,  sqlStringType));

      createTableIfNotExist(layerTableName, columns);
    }
#endif
  }

  void ConvDatabase::insertPerfResults(const ConvParam convParam, const ConvType convType, const int batchSize,
				       const size_t workspaceSize,
				       const std::vector<cudnnConvolutionGenericAlgoPerf_t> perfs) {
#ifdef UCUDNN_USE_SQLITE

    const size_t cudnnVersion = cudnnGetVersion();
    const std::string gpuName = getGPUName();

    const std::string convParamHash = convParam.databaseHash();

    for(auto i = perfs.begin(); i != perfs.end(); i++) {
      DatabaseEntry entry;
      entry.setColumn(cName_convParam,     convParamHash);
      entry.setColumn(cName_convType,      convTypeToString(convType));
      entry.setColumn(cName_batchSize,     batchSize);
      entry.setColumn(cName_workspaceSize, workspaceSize);
      entry.setColumn(cName_cudnnVersion,  cudnnVersion);
      entry.setColumn(cName_gpuName,       gpuName);

      entry.setColumn(cName_perfRank,        (int) std::distance(perfs.begin(), i));
      entry.setColumn(cName_perfAlgo,        (*i).algo);
      entry.setColumn(cName_perfTime,        (*i).time);
      entry.setColumn(cName_perfMemory,      (*i).memory);
      entry.setColumn(cName_perfDeterminism, (*i).determinism == CUDNN_DETERMINISTIC);
#if CUDNN_HAS_MATHTYPE
      entry.setColumn(cName_perfMathType,    (*i).mathType == CUDNN_TENSOR_OP_MATH);
#endif

      insert(perfTableName, entry);
    }
#endif
  }

  void ConvDatabase::setLayerParams(const OptCache &optCache) {
#ifdef UCUDNN_USE_SQLITE
    deleteEntries(layerTableName);

    for(const auto i : optCache.getParameters()) {
      const LayerId layerId = i.first.first;
      const ConvType convType = i.first.second;
      const ConvParam convParam = i.second;
      DatabaseEntry entry;
      entry.setColumn(cName_layerId, (size_t) layerId);
      entry.setColumn(cName_convType, convTypeToString(convType));
      entry.setColumn(cName_convParam, convParam.databaseHash());
      insert(layerTableName, entry);
    }
#endif
  }

  std::vector<cudnnConvolutionGenericAlgoPerf_t>
  ConvDatabase::selectPerfResults(const ConvParam convParam, const ConvType convType, const int batchSize, const size_t workspaceSize) {

    const size_t cudnnVersion = cudnnGetVersion();
    const std::string gpuName = getGPUName();

    DatabaseEntry condition;
    condition.setColumn(cName_convParam,     convParam.databaseHash());
    condition.setColumn(cName_convType,      convTypeToString(convType));
    condition.setColumn(cName_batchSize,     batchSize);
    condition.setColumn(cName_cudnnVersion,  cudnnVersion);
    condition.setColumn(cName_gpuName,       gpuName);

    DatabaseEntry columns;
    columns.setColumnAndType(cName_workspaceSize,   INT);
    columns.setColumnAndType(cName_perfRank,        INT);
    columns.setColumnAndType(cName_perfAlgo,        INT);
    columns.setColumnAndType(cName_perfTime,        DOUBLE);
    columns.setColumnAndType(cName_perfMemory,      INT);
    columns.setColumnAndType(cName_perfDeterminism, BOOL);
#if CUDNN_HAS_MATHTYPE
    columns.setColumnAndType(cName_perfMathType,    BOOL);
#endif

    const std::vector<DatabaseEntry> entries = selectEquals(perfTableName, columns, condition);

    std::vector<cudnnConvolutionGenericAlgoPerf_t> perfs;

    size_t benchmarkedWorkspace = 0;
    for(auto i = entries.begin(); i != entries.end(); i++) {
      const size_t ws = (*i).getColumn(cName_workspaceSize);
      if(ws <= workspaceSize && ws > benchmarkedWorkspace)
	benchmarkedWorkspace = ws;
    }

    std::vector<DatabaseEntry> perfEntries(entries.size());
    {
      const auto perfEntriesEnd = std::copy_if(entries.begin(), entries.end(), perfEntries.begin(),
					       [&benchmarkedWorkspace](DatabaseEntry entry) {
						 return (size_t) entry.getColumn(cName_workspaceSize)
						 == benchmarkedWorkspace; });
      perfEntries.resize(std::distance(perfEntries.begin(), perfEntriesEnd));
    }

    std::sort(perfEntries.begin(), perfEntries.end(),
	      [](DatabaseEntry e1, DatabaseEntry e2) {
		return (int) e1.getColumn(cName_perfRank) < (int) e2.getColumn(cName_perfRank); });

    for(auto i = perfEntries.begin(); i != perfEntries.end(); i++) {
      cudnnConvolutionGenericAlgoPerf_t perf;
      perf.status      = CUDNN_STATUS_SUCCESS;
      perf.algo        = (*i).getColumn(cName_perfAlgo);
      perf.time        = (float) (*i).getColumn(cName_perfTime);
      perf.memory      = (*i).getColumn(cName_perfMemory);
      perf.determinism = (bool) (*i).getColumn(cName_perfDeterminism) ? CUDNN_DETERMINISTIC : CUDNN_NON_DETERMINISTIC;
#if CUDNN_HAS_MATHTYPE
      perf.mathType    = (bool) (*i).getColumn(cName_perfMathType)    ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
#endif
      perfs.push_back(perf);
    }

    return perfs;
  }

  void ConvDatabase::query(const std::string sql) {
#ifdef UCUDNN_USE_SQLITE
    std::cerr << sql << std::endl;
    UCUDNN_SQLITE_CHECK(sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, nullptr));
#endif
  }

  void ConvDatabase::deleteEntries(const std::string tableName) {
#ifdef UCUDNN_USE_SQLITE
    query("DELETE FROM "+tableName);
#endif
  }

  void ConvDatabase::createTableIfNotExist(const std::string tableName,
					   const std::vector<std::pair<std::string, std::string> > columns) {
#ifdef UCUDNN_USE_SQLITE
    std::string sql = "CREATE TABLE IF NOT EXISTS " + tableName;
    sql += " (";
    for(auto i = columns.begin(); i != columns.end(); i++) {
      if(i != columns.begin())
	sql += ",";
      sql += (*i).first + " " + (*i).second;
    }
    sql += ")";

    query(sql);
#endif
  }

  void ConvDatabase::insert(const std::string tableName, const DatabaseEntry entry) {
#ifdef UCUDNN_USE_SQLITE
    const auto keys = entry.keys();

    std::string sql = "INSERT INTO " + tableName + " (" + joinToString(keys) + ") VALUES (";
    for(auto i = keys.begin(); i != keys.end(); i++) {
      if(i != keys.begin())
	sql += ", ";
      sql += entry.getColumn(*i).toString();
    }
    sql += ")";

    query(sql);
#endif
  }

  std::vector<DatabaseEntry> ConvDatabase::selectEquals(const std::string tableName,
							const DatabaseEntry columns, const DatabaseEntry condition) {
#ifdef UCUDNN_USE_SQLITE

    std::string sql = "SELECT " + joinToString(columns.keys()) + " FROM " + tableName + " WHERE ";
    {
      const auto keys = condition.keys();
      for(auto i = keys.begin(); i != keys.end(); i++) {
	if(i != keys.begin())
	  sql += " AND ";
	sql += (*i) + "=" + condition.getColumn(*i).toString();
      }
    }

    sqlite3_stmt *stmt = nullptr;
    UCUDNN_SQLITE_CHECK(sqlite3_prepare_v2(db_,
					   sql.c_str(),
					   sql.size()+1, &stmt, NULL));

    std::vector<DatabaseEntry> ret;
    const auto keys = columns.keys();

    while(sqlite3_step(stmt) == SQLITE_ROW) {
      const int columnId = sqlite3_column_int(stmt, 0);
      DatabaseEntry entry;
      for(auto i = keys.begin(); i != keys.end(); i++) {
	const char *str = (char *) sqlite3_column_text(stmt, std::distance(keys.begin(), i));
	entry.setColumn(*i, std::string(str), columns.getColumn(*i).type());
      }
      ret.push_back(entry);
    }

    UCUDNN_SQLITE_CHECK(sqlite3_finalize(stmt));

    return ret;

#endif
  }

}
