/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/abi.h"

#if defined(PLATFORM_WINDOWS)
#include <cstring>
#else
#include <cxxabi.h>
#include <cstdlib>
#include <memory>
#endif

#include <string>

namespace tensorflow {
namespace port {

std::string MaybeAbiDemangle(const char* name) {
#if defined(PLATFORM_WINDOWS)
  const char* p = std::strchr(name, ' ');
  return p ? p + 1 : name;
#else
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}

}  // namespace port
}  // namespace tensorflow
