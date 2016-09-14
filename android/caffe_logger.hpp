#ifndef __CAFFE_CORE_COMMON_LOGGER_H__
#define __CAFFE_CORE_COMMON_LOGGER_H__

#pragma warning ( disable : 4127 )

//#define __ANDROID__

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#elif defined(__ANDROID__) || defined(android)
#include <android/log.h>
#include <assert.h>
#else
#include <inttypes.h>
#endif

#include <time.h>

#ifdef __cplusplus
extern "C"{
#endif
double gettimeofday_sec()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec*1e-6;
};


#define CAFFE_LOG_LEVEL_ALL     65535
#define CAFFE_LOG_LEVEL_ERROR   0
#define CAFFE_LOG_LEVEL_WARNING 10000
#define CAFFE_LOG_LEVEL_INFO    20000
#define CAFFE_LOG_LEVEL_DEBUG   30000
#define CAFFE_LOG_LEVEL_VERBOSE 65535
#define CAFFE_DEBUG 1
#define CAFFE_LOG_LEVEL CAFFE_LOG_LEVEL_ALL

#if defined(_WIN32) || defined(_WIN64)
#  define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
  //#  define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#  define __FILENAME__ __FILE__
#endif

#if !defined(LOG_TAG)
#  if defined(_WIN32) || defined(_WIN64)
#    define LOG_TAG __FILENAME__
#  else
#    define LOG_TAG __FILENAME__
#  endif
#endif

#if defined(__ENABLE_ASSERT__)
#  define CAFFE_ASSERT(_cond) assert(_cond)
#else
#  define CAFFE_ASSERT(_cond) assert(1)
#endif

#  define TP_STR_HELPER(x) #x
#  define TP_STR(x) TP_STR_HELPER(x)

#if defined(_WIN32) || defined(_WIN64)
#  define CAFFE_Log(FP,LEVEL,FMT,...) \
    fprintf(FP,"[%s] %s(%4d):%s:" ## FMT ## "\n" ,LEVEL , LOG_TAG ,__LINE__ , __FUNCTION__ , __VA_ARGS__)
#  define CAFFE_LogSelf(FP,LEVEL, TAG, LINE, FUNCTION, FMT,...) \
    fprintf(FP,"[%s] %s(%4d):%s:" ## FMT ## "\n" ,LEVEL , TAG , LINE, FUNCTION , __VA_ARGS__)
#  if CAFFE_DEBUG
#    define CAFFE_DebugLog(FMT,...)       if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_DEBUG)   CAFFE_Log(stdout, "Debug  :",FMT, __VA_ARGS__)
#    define CAFFE_ErrLog(FMT,...)         if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_ERROR)   CAFFE_Log(stderr, "Err    :",FMT, __VA_ARGS__)
#    define CAFFE_WarnLog(FMT,...)        if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_WARNING) CAFFE_Log(stdout, "Warn   :",FMT, __VA_ARGS__)
#    define CAFFE_InfoLog(FMT,...)        if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_INFO)    CAFFE_Log(stdout, "Info   :",FMT, __VA_ARGS__)
#    define CAFFE_VerboseLog(FMT,...)     if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_VERBOSE) CAFFE_Log(stdout, "Verbose:",FMT, __VA_ARGS__)
#    define CAFFE_AssertLog(_cond, FMT, ...)  CAFFE_ErrLog(FMT,__VA_ARGS__);\
                                            CAFFE_ASSERT(_cond)
#  else
#    define CAFFE_DebugLog(FMT,...)          do{}while(0,0)
#    define CAFFE_WarnLog(FMT,...)           do{}while(0,0)
#    define CAFFE_ErrLog(FMT,...)            CAFFE_Log(stderr, "Err    :",FMT, __VA_ARGS__)
#    define CAFFE_InfoLog(FMT,...)           do{}while(0,0)
#    define CAFFE_VerboseLog(FMT,...)        do{}while(0,0)
#    define CAFFE_AssertLog(_cond, FMT,...)  do{}while(0,0)
#  endif
#elif defined(__APPLE__)
#	define CAFFE_Log(FP, LEVEL,args...) \
		fprintf(FP, "[%s] %s(%4d):%s:" ,LEVEL , LOG_TAG ,__LINE__ , __FUNCTION__); printf(args); printf("\n")
#	define CAFFE_LogSelf(LEVEL, TAG, LINE, FUNCTION, args...) \
		fprintf(FP, "[%s] %s(%4d):%s:" ,LEVEL , TAG , LINE, FUNCTION); printf(args); printf("\n")
#  if CAFFE_DEBUG
#    define CAFFE_DebugLog(...)          if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_DEBUG)   CAFFE_Log(stdout, "Debug  :" ,__VA_ARGS__)
#    define CAFFE_ErrLog(...)            if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_ERROR)   CAFFE_Log(stderr, "Error  :" ,__VA_ARGS__)
#    define CAFFE_WarnLog(...)           if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_WARNING) CAFFE_Log(stdout, "Warn   :" ,__VA_ARGS__)
#    define CAFFE_InfoLog(...)           if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_INFO)    CAFFE_Log(stdout, "Info   :" ,__VA_ARGS__)
#    define CAFFE_VerboseLog(...)        if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_VERBOSE) CAFFE_Log(stdout, "Verbose:" ,__VA_ARGS__)
#    define CAFFE_AssertLog(_cond, ...)  CAFFE_ErrLog(__VA_ARGS__); \
                                       CAFFE_ASSERT(_cond)

#  else
#    define CAFFE_DebugLog(...)              do{}while(0,0)
#    define CAFFE_WarnLog(...)               do{}while(0,0)
#    define CAFFE_ErrLog(...)                if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_ERROR)   CAFFE_Log(stderr, "Warn :" ,__VA_ARGS__)
#    define CAFFE_InfoLog(FMT,...)           do{}while(0,0)
#    define CAFFE_VerboseLog(...)            do{}while(0,0)
#    define CAFFE_AssertLog(_cond, args...)  CAFFE_ErrLog(args); CAFFE_ASSERT(_cond)
#  endif
#else
#  if CAFFE_DEBUG
#    define CAFFE_DebugLog(...)          if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_DEBUG)   __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG TP_STR(__LINE__), __VA_ARGS__)
#    define CAFFE_ErrLog(...)            if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_ERROR)   __android_log_print(ANDROID_LOG_ERROR, LOG_TAG TP_STR(__LINE__), __VA_ARGS__)
#    define CAFFE_InfoLog(...)           if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_INFO)    __android_log_print(ANDROID_LOG_INFO,  LOG_TAG TP_STR(__LINE__), __VA_ARGS__)
#    define CAFFE_WarnLog(...)           if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_WARNING) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG TP_STR(__LINE__), __VA_ARGS__)
#    define CAFFE_VerboseLog(...)        do{}while(0,0)
#    define CAFFE_AssertLog(_cond, ...)  CAFFE_ErrLog(__VA_ARGS__);\
                                       CAFFE_ASSERT(_cond)
#  else
#    define CAFFE_DebugLog(...)          do{}while(0,0)
#    define CAFFE_WarnLog(...)           do{}while(0,0)
#    define CAFFE_ErrLog(...)            if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_ERROR)   __android_log_print(ANDROID_LOG_INFO,  LOG_TAG TP_STR(__LINE__), __VA_ARGS__)
#    define CAFFE_InfoLog(...)           if (CAFFE_LOG_LEVEL >= CAFFE_LOG_LEVEL_WARNING) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG TP_STR(__LINE__), __VA_ARGS__)
#    define CAFFE_VerboseLog(...)        do{}while(0,0)
#    define CAFFE_AssertLog(_cond, ...)  CAFFE_ErrLog(__VA_ARGS__); \
                                       CAFFE_ASSERT(_cond)
#  endif
#endif

#if defined(_WIN32) || defined(_WIN64)
#  define CAFFE_DBG(FMT,...)             CAFFE_DebugLog(FMT,__VA_ARGS__)
#  define CAFFE_WARN(FMT,...)            CAFFE_WarnLog(FMT,__VA_ARGS__)
#  define CAFFE_ERR(FMT,...)             CAFFE_ErrLog(FMT,__VA_ARGS__)
#  define CAFFE_INFO(FMT,...)            CAFFE_InfoLog(FMT,__VA_ARGS__)
#  define CAFFE_VERB(FMT,...)            CAFFE_VerboseLog(FMT,__VA_ARGS__)
#  define CAFFE_ASET(_cond, FMT,...)     CAFFE_AssertLog(_cond,FMT,__VA_ARGS__)
#else
#  define CAFFE_DBG(...)                 CAFFE_DebugLog(__VA_ARGS__)
#  define CAFFE_WARN(...)                CAFFE_WarnLog(__VA_ARGS__)
#  define CAFFE_ERR(...)                 CAFFE_ErrLog(__VA_ARGS__)
#  define CAFFE_INFO(...)                CAFFE_InfoLog(__VA_ARGS__)
#  define CAFFE_VERB(...)                CAFFE_VerboseLog(__VA_ARGS__)
#  define CAFFE_ASET(_cond, ...)         CAFFE_AssertLog(_cond,__VA_ARGS__)
#endif


#if !defined(CAFFE_StartLog) && !defined(CAFFE_EndLog)
#  if CAFFE_DEBUG
#    define CAFFE_StartLog() \
			double clock_function_start = gettimeofday_sec();\
			CAFFE_INFO("%s started..",__FUNCTION__)
#    define CAFFE_EndLog() \
            CAFFE_INFO("%s endl..[%f]", __FUNCTION__, (double)(gettimeofday_sec() - clock_function_start))
#  else
#    define CAFFE_StartLog() do{}while(0,0)
#    define CAFFE_EndLog() do{}while(0,0)
#  endif
#endif

#ifdef __cplusplus
}
#endif
#endif

