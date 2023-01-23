// Copyright 2022 PingCAP, Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <IO/BufferWithOwnMemory.h>
#include <IO/CompressedReadBufferBase.h>
#include <IO/ReadBuffer.h>

namespace LZ4
{
struct StreamStatistics;
struct PerformanceStatistics;
} // namespace LZ4
namespace DB
{
class FasterCompressedReadBuffer : public CompressedReadBufferBase<false>
    , public BufferWithOwnMemory<ReadBuffer>
{
private:
    using Base = CompressedReadBufferBase<false>;
    using Self = FasterCompressedReadBuffer;
    bool nextImpl() override;
    void decompress(char * to, size_t size_decompressed, size_t size_compressed_without_checksum);

public:
    static std::unique_ptr<Self> genFasterCompressedReadBuffer();
    explicit FasterCompressedReadBuffer(ReadBuffer & in_, LZ4::PerformanceStatistics & statistics_);

    size_t readBig(char * to, size_t n) override;

    /// The compressed size of the current block.
    size_t getSizeCompressed() const { return size_compressed; }

private:
    size_t size_compressed = 0;
    LZ4::PerformanceStatistics & statistics;
};

} // namespace DB
