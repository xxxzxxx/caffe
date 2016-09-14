#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C"{
#endif

#define JNIFunctionDefine(_Package,_Cls, _Func) Java_## _Package ##_## _Cls ##_## _Func
#ifdef __cplusplus
}
#endif

