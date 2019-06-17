// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "local_filesystem.h"

class UnmapFileParam {
 public:
  void* addr;
  size_t len;
  int fd;
  std::string file_path;
};

static void UnmapFile(void* param) noexcept {
  std::unique_ptr<UnmapFileParam> p(reinterpret_cast<UnmapFileParam*>(param));
  int ret = munmap(p->addr, p->len);
  if (ret != 0) {
    ReportSystemError("munmap", p->file_path);
    return;
  }
  if (close(p->fd) != 0) {
    ReportSystemError("close", p->file_path);
    return;
  }
}

void ReadFileAsString(const ORTCHAR_T* fname, void*& p, size_t& len, Callback& deleter) {
  if (!fname) {
    throw std::runtime_error("ReadFileAsString: 'fname' cannot be NULL");
  }

  deleter.f = nullptr;
  deleter.param = nullptr;
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    return ReportSystemError("open", fname);
  }
  struct stat stbuf;
  if (fstat(fd, &stbuf) != 0) {
    return ReportSystemError("fstat", fname);
  }

  if (!S_ISREG(stbuf.st_mode)) {
    throw std::runtime_error("ReadFileAsString: input is not a regular file");
  }
  // TODO:check overflow
  len = static_cast<size_t>(stbuf.st_size);

  if (len == 0) {
    p = nullptr;
  } else {
    p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      // TODO: assert(close(fd) == 0);
      ReportSystemError("mmap", fname);
    } else {
      // leave the file open
      deleter.f = UnmapFile;
      deleter.param = new UnmapFileParam{p, len, fd, fname};
      p = reinterpret_cast<char*>(p);
    }
  }

  // assert(close(fd) == 0);
}
