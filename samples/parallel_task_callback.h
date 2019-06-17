#pragma once

#include "Callback.h"

#include <mutex>
#include <atomic>
#include <functional>

class ParallelTaskCallback {
 private:
  std::function<void()> c_;
  std::atomic<int> finished_;

 public:
  ParallelTaskCallback(int task_count, const std::function<void()>& c) : c_(c), finished_(task_count) {}

  void Finish() {
    const bool is_done = --finished_ == 0;
    if (is_done) {
      std::function<void()> t = c_;
      delete this;
      t();
    }
  }
};