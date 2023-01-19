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

#include <IO/CompressedStream.h>
#include <IO/FasterCompressedReadBuffer.h>
#include <IO/LZ4_decompress_faster.h>
#include <common/types.h>


namespace DB
{
FasterCompressedReadBuffer::FasterCompressedReadBuffer(ReadBuffer & in_, LZ4::StreamStatistics & statistics_)
    : Base(&in_)
    , BufferWithOwnMemory<ReadBuffer>(0)
    , statistics(statistics_)
{}

bool FasterCompressedReadBuffer::nextImpl()
{
    size_t size_decompressed;
    size_t size_compressed_without_checksum;
    size_compressed = this->readCompressedData(size_decompressed, size_compressed_without_checksum);
    if (!size_compressed)
        return false;

    memory.resize(size_decompressed);
    working_buffer = Buffer(&memory[0], &memory[size_decompressed]);

    this->decompress(working_buffer.begin(), size_decompressed, size_compressed_without_checksum);

    return true;
}

void FasterCompressedReadBuffer::decompress(char * to, size_t size_decompressed, size_t size_compressed_without_checksum)
{
    UInt8 method = compressed_buffer[0]; /// See CompressedWriteBuffer.h

    if (method == static_cast<UInt8>(CompressionMethodByte::LZ4))
    {
        LZ4::decompress(compressed_buffer + COMPRESSED_BLOCK_HEADER_SIZE, to, size_compressed_without_checksum - COMPRESSED_BLOCK_HEADER_SIZE, size_decompressed, statistics);

        if (unlikely(LZ4_decompress_safe(compressed_buffer + COMPRESSED_BLOCK_HEADER_SIZE, to, size_compressed_without_checksum - COMPRESSED_BLOCK_HEADER_SIZE, size_decompressed) < 0))
            throw Exception("Cannot LZ4_decompress_safe", ErrorCodes::CANNOT_DECOMPRESS);
    }
    else if (method == static_cast<UInt8>(CompressionMethodByte::ZSTD))
    {
        size_t res = ZSTD_decompress(to, size_decompressed, compressed_buffer + COMPRESSED_BLOCK_HEADER_SIZE, size_compressed_without_checksum - COMPRESSED_BLOCK_HEADER_SIZE);

        if (ZSTD_isError(res))
            throw Exception("Cannot ZSTD_decompress: " + std::string(ZSTD_getErrorName(res)), ErrorCodes::CANNOT_DECOMPRESS);
    }
    else if (method == static_cast<UInt8>(CompressionMethodByte::NONE))
    {
        memcpy(to, &compressed_buffer[COMPRESSED_BLOCK_HEADER_SIZE], size_decompressed);
    }
    else
        throw Exception("Unknown compression method: " + toString(method), ErrorCodes::UNKNOWN_COMPRESSION_METHOD);
}


size_t FasterCompressedReadBuffer::readBig(char * to, size_t n)
{
    size_t bytes_read = 0;

    /// If there are unread bytes in the buffer, then we copy necessary to `to`.
    if (pos < working_buffer.end())
        bytes_read += read(to, std::min(static_cast<size_t>(working_buffer.end() - pos), n));

    /// If you need to read more - we will, if possible, uncompress at once to `to`.
    while (bytes_read < n)
    {
        size_t size_decompressed;
        size_t size_compressed_without_checksum;

        if (!this->readCompressedData(size_decompressed, size_compressed_without_checksum))
            return bytes_read;

        /// If the decompressed block is placed entirely where it needs to be copied.
        if (size_decompressed <= n - bytes_read)
        {
            this->decompress(to + bytes_read, size_decompressed, size_compressed_without_checksum);
            bytes_read += size_decompressed;
            bytes += size_decompressed;
        }
        else
        {
            bytes += offset();
            memory.resize(size_decompressed);
            working_buffer = Buffer(&memory[0], &memory[size_decompressed]);
            pos = working_buffer.begin();

            this->decompress(working_buffer.begin(), size_decompressed, size_compressed_without_checksum);

            bytes_read += read(to + bytes_read, n - bytes_read);
            break;
        }
    }

    return bytes_read;
}

} // namespace DB
