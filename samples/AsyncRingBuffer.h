#pragma once
#include <iostream>



template <typename T>
class SyncQueue {
  std::vector<T>& tasks;
  size_t current_task_id = 0;
  std::mutex m;
  DataProcessing* p;

 public:
  SyncQueue(std::vector<T>& input_tasks, DataProcessing* p1) : tasks(input_tasks), p(p1) {}

  // return true if eof
  bool Next(void* output_data) {
    T* data = nullptr;
    bool is_eof = false;
    {
      std::lock_guard<std::mutex> g(m);
      if (current_task_id >= tasks.size()) {
        is_eof = true;
      }
      data = &tasks[current_task_id];
      ++current_task_id;
    }
    if (is_eof) {
      delete this;
      return false;
    }
    (*p)(data, output_data);
    return true;
  }
};

class AsyncRingBuffer {
 public:
  enum class BufferState { EMPTY, FILLING };
  uint8_t* buffer;
  size_t item_size;
  size_t capacity;
  std::vector<BufferState> buffer_state;
  size_t read_index;
  size_t write_index;
  size_t batch_size;
  size_t parallelism = 8;
  size_t current_running_downloders = 0;
  std::mutex m;
  GThreadPool* threadpool;

  SyncQueue<std::string>& downloader;

  AsyncRingBuffer(
  size_t item_size1,
  size_t capacity1,GThreadPool* threadpool1, SyncQueue<std::string>& downloader1)
      : item_size(item_size1), capacity(capacity1), buffer_state(capacity1, BufferState::EMPTY), read_index(0), write_index(0),
        threadpool(threadpool1),
      downloader(downloader1) {
    buffer = new uint8_t[item_size * capacity];
  }

  void Download(void* dest) {
    bool has_data = downloader.Next(dest);
    if (has_data) {
      StartDownloadTasks();
    } else {
      // TODO: notify eof
    }
  }

  bool StartDownloadTasks() {
    // search empty slots, launch a download task for each of them
    std::vector<void*> tasks_to_launch;
    {
      std::lock_guard<std::mutex> g(m);
      for (size_t i = 0; i != capacity; ++i) {
        if (current_running_downloders + tasks_to_launch.size() >= parallelism) break;
        size_t index = (write_index + i) % capacity;
        if (buffer_state[i] == BufferState::EMPTY) {
          tasks_to_launch.push_back(buffer + index * item_size);
        }
      }
    }
    class DownloadTask : public RunnableTask {
     public:
      AsyncRingBuffer* requester;
      void* dest;
      DownloadTask(AsyncRingBuffer* r, void* d) : requester(r), dest(d) {}

      void operator()() override {
        AsyncRingBuffer* r = requester;
        void* d = dest;
        delete this;
        r->Download(d);
      }
    };
    GError* err = nullptr;
    for (void* p : tasks_to_launch) {
      g_thread_pool_push(threadpool, new DownloadTask(this, p), &err);
      if (err != nullptr) {
        fprintf(stderr, "Unable to create thread pool: %s\n", err->message);
        g_error_free(err);
        return false;
      }
    }
    return true;
  }
};