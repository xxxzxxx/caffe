#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include "caffe_jni.hpp"
#include "caffe_logger.hpp"

#ifdef USE_EIGEN
#include <omp.h>
#else
#include <cblas.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define JNIFunctionDefineDefaultPackage(_Cls, _Func) JNIFunctionDefine(\
		com_xxxzxxx_caffe_android_lib\
		, _Cls\
		, _Func\
		)

using std::string;
using std::vector;
using caffe::CaffeMobile;

static string jstring2string(JNIEnv *env, jstring jstr) {
  CAFFE_StartLog();
  const char *cstr = env->GetStringUTFChars(jstr, 0);
  string str(cstr);
  env->ReleaseStringUTFChars(jstr, cstr);
  CAFFE_EndLog();
  return str;
}

/**
 * NOTE: byte[] buf = str.getBytes("US-ASCII")
 */
static string bytes2string(JNIEnv *env, jbyteArray buf) 
{
  CAFFE_StartLog();
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  string s((char *)ptr, env->GetArrayLength(buf));
  env->ReleaseByteArrayElements(buf, ptr, 0);
  CAFFE_EndLog();
  return s;
}

static cv::Mat imgbuf2mat(JNIEnv *env, jbyteArray buf, int width, int height) 
{
  CAFFE_StartLog();
  cv::Mat result;
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  const int byte_length = env->GetArrayLength(buf);
  const int yuv_length = height + height / 2 * width;
  const int rgb_length = height * width * 3;
  const int rgba_length = height * width * 4;
  //YUV
  if (byte_length == yuv_length)
  {
    CAFFE_DBG("type YUV");
    result = cv::Mat(height + height / 2, width, CV_8UC1, (unsigned char *)ptr);
    cv::cvtColor(result, result, CV_YUV2RGBA_NV21);
    env->ReleaseByteArrayElements(buf, ptr, 0);
  }
  //RGB
  else if (byte_length == rgb_length)
  {
    CAFFE_DBG("type RGB");
    result = cv::Mat(height, width, CV_8UC3, (unsigned char *)ptr);
    cv::cvtColor(result, result, CV_RGB2RGBA);
    env->ReleaseByteArrayElements(buf, ptr, 0);
  }
  //RGBA
  else if (byte_length == rgba_length)
  {
    CAFFE_DBG("type RGBA");
    result = cv::Mat(height, width, CV_8UC4, (unsigned char *)ptr);
  }
  else
  {
    CAFFE_DBG("type unknown");
    throw -1;
  }
  CAFFE_EndLog();
  return result;
}
static cv::Mat getImage(JNIEnv *env, jbyteArray buf, int width, int height) {
  CAFFE_StartLog();
  cv::Mat result;
  if (width == 0 && height == 0) 
  {
    result = cv::imread(bytes2string(env, buf), -1);
  }
  else
  {
    result = imgbuf2mat(env, buf, width, height);
  }
  CAFFE_EndLog();
  return result;
}

JNIEXPORT jlong JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,CreateCaffeMobile)
(JNIEnv *env,jobject thiz) 
{
  CAFFE_StartLog();
  auto caffe_mobile = std::shared_ptr<CaffeMobile>(new CaffeMobile());
  auto instance_id = CaffeMobile::PutStoreInstance(caffe_mobile);
  CAFFE_EndLog();
  return instance_id;
}

JNIEXPORT void JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,ReleaseCaffeMobile)
(JNIEnv *env,jobject thiz,jlong instance_id) 
{
  CAFFE_StartLog();
  CaffeMobile::EraseStoredInstance(instance_id);
  CAFFE_EndLog();
}

JNIEXPORT void JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,SetNumThreads)
(JNIEnv *env,jobject thiz,jint numThreads) 
{
  CAFFE_StartLog();
  int num_threads = numThreads;
 #ifdef USE_EIGEN
   omp_set_num_threads(num_threads);
 #else
   openblas_set_num_threads(num_threads);
 #endif
  CAFFE_EndLog();
}

JNIEXPORT jint JNICALL JNIFunctionDefineDefaultPackage(CaffeMobile,LoadModel)
(JNIEnv *env, jobject thiz, jlong instance_id, jstring modelPath, jstring weightsPath) 
{
  CAFFE_StartLog();
  jint result = 0;
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  if (caffe_mobile == nullptr)
  {
    result = -1;
  }
  else
  {
    auto stringModelPath = jstring2string(env, modelPath);
    auto stringWeightsPath = jstring2string(env, weightsPath);
    caffe_mobile->LoadModule(stringModelPath,stringWeightsPath);
  }
  CAFFE_EndLog();
  return result;
}

JNIEXPORT jint JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,SetMeanWithMeanFile)
(JNIEnv *env, jobject thiz, jlong instance_id, jstring meanFile) 
{
  CAFFE_StartLog();
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  jint result = 0;
  if (caffe_mobile == nullptr)
  {
    result = -1;
  }
  else
  {
    caffe_mobile->SetMean(jstring2string(env, meanFile));
  }
  CAFFE_EndLog();
  return result;
}

JNIEXPORT jint JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,SetMeanWithMeanValues)
(JNIEnv *env, jobject thiz, jlong instance_id, jfloatArray meanValues) 
{
  CAFFE_StartLog();
  jint result = 0;
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  if (caffe_mobile == nullptr)
  {
    result = -1;
  }
  else
  {
    int num_channels = env->GetArrayLength(meanValues);
    jfloat *ptr = env->GetFloatArrayElements(meanValues, 0);
    vector<float> mean_values(ptr, ptr + num_channels);
    caffe_mobile->SetMean(mean_values);
  }
  CAFFE_EndLog();
  return result;
}

JNIEXPORT jint JNICALL JNIFunctionDefineDefaultPackage(CaffeMobile,SetScale)
(JNIEnv *env, jobject thiz, jlong instance_id, jfloat scale) 
{
  CAFFE_StartLog();
  jint result = 0;
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  if (caffe_mobile == nullptr)
  {
    result = -1;
  }
  else
  {
    caffe_mobile->SetScale(scale);
  }
  CAFFE_EndLog();
  return result;
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jfloatArray JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,GetConfidenceScore)
(JNIEnv *env, jobject thiz, jlong instance_id, jbyteArray buf, jint width, jint height) 
{
  CAFFE_StartLog();
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  jfloatArray result = NULL;
  if (caffe_mobile == nullptr)
  {
    vector<float> conf_score = caffe_mobile->GetConfidenceScore(getImage(env, buf, width, height));
    result = env->NewFloatArray(conf_score.size());
    if (result != NULL) 
    {
      // move from the temp structure to the java structure
      env->SetFloatArrayRegion(result, 0, conf_score.size(), &conf_score[0]);
    }
  }
  CAFFE_EndLog();
  return result;
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jintArray JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,PredictImage)
(JNIEnv *env, jobject thiz, jlong instance_id, jbyteArray buf, jint width, jint height,jint k) 
{
  CAFFE_StartLog();
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  jintArray result = NULL;
  if (caffe_mobile != nullptr)
  {
    vector<int> top_k = caffe_mobile->PredictTopK(getImage(env, buf, width, height), k);
    result = env->NewIntArray(k);
    if (result != NULL)
    {
      // move from the temp structure to the java structure
      env->SetIntArrayRegion(result, 0, k, &top_k[0]);
    }
  }
  CAFFE_EndLog();
  return result;
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jobjectArray JNICALL
JNIFunctionDefineDefaultPackage(CaffeMobile,ExtractFeatures)
(JNIEnv *env, jobject thiz, jlong instance_id, jbyteArray buf, jint width, jint height,jstring blobNames) 
{
  CAFFE_StartLog();
  auto caffe_mobile = CaffeMobile::FindStoredInstance(instance_id);
  jobjectArray array2D = NULL;
  if (caffe_mobile != nullptr)
  {
    vector<vector<float>> features = caffe_mobile->ExtractFeatures(getImage(env, buf, width, height), jstring2string(env, blobNames));
    array2D =env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
    for (size_t i = 0; i < features.size(); ++i) 
    {
      jfloatArray array1D = env->NewFloatArray(features[i].size());
      if (array1D == NULL) 
      {
        return NULL; /* out of memory error thrown */
      }
      // move from the temp structure to the java structure
      env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
      env->SetObjectArrayElement(array2D, i, array1D);
    }
  }
  CAFFE_EndLog();
  return array2D;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  CAFFE_StartLog();
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }

#if defined(USE_GLOG)
  FLAGS_redirecttologcat = true;
  FLAGS_android_logcat_tag = "caffe_jni";
#endif
  CAFFE_EndLog();
  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
