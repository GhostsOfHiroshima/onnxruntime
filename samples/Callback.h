// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct Callback {
  void (*f)(void* param);
  void* param;
};
