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

#include <Common/TiFlashException.h>
#include <Common/TiFlashMetrics.h>
#include <Flash/Coprocessor/CHBlockChunkCodec.h>
#include <Flash/Mpp/HashBaseWriterHelper.h>
#include <Flash/Mpp/HashPartitionWriterV1.h>
#include <Flash/Mpp/MPPTunnelSet.h>
#include <kvproto/mpp.pb.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>

#include "Common/Exception.h"
#include "Common/Stopwatch.h"
#include "Flash/Coprocessor/CHBlockChunkCodecStream.h"
#include "Flash/Coprocessor/CompressCHBlockChunkCodecStream.h"
#include "Flash/Coprocessor/CompressedCHBlockChunkCodec.h"
#include "Flash/Mpp/MppVersion.h"
#include "IO/CompressedStream.h"
#include "IO/CompressionSettings.h"
#include "common/logger_useful.h"
#include "ext/scope_guard.h"

namespace DB
{

CompressionMethod ToInternalCompressionMethod(mpp::CompressionMode compression_mode)
{
    switch (compression_mode)
    {
    case mpp::CompressionMode::NONE:
        return CompressionMethod::NONE;
    case mpp::CompressionMode::FAST:
        return CompressionMethod::LZ4; // use LZ4 method as fast mode
    case mpp::CompressionMode::HIGH_COMPRESSION:
        return CompressionMethod::ZSTD; // use ZSTD method as HC mode
    default:
        return CompressionMethod::NONE;
    }
}

template <class ExchangeWriterPtr>
HashPartitionWriterImplV1<ExchangeWriterPtr>::HashPartitionWriterImplV1(
    ExchangeWriterPtr writer_,
    std::vector<Int64> partition_col_ids_,
    TiDB::TiDBCollators collators_,
    Int64 partition_batch_limit_,
    bool should_send_exec_summary_at_last_,
    DAGContext & dag_context_,
    mpp::CompressionMode compression_mode_)
    : DAGResponseWriter(/*records_per_chunk=*/-1, dag_context_)
    , partition_num(writer_->getPartitionNum())
    , partition_batch_limit(partition_batch_limit_ * partition_num)
    , should_send_exec_summary_at_last(should_send_exec_summary_at_last_)
    , writer(writer_)
    , partition_col_ids(std::move(partition_col_ids_))
    , collators(std::move(collators_))
    , compression_mode(compression_mode_)
{
    assert(dag_context.getMPPTaskMeta().mpp_version() > 0);

    rows_in_blocks = 0;
    RUNTIME_CHECK(partition_num > 0);
    RUNTIME_CHECK(dag_context.encode_type == tipb::EncodeType::TypeCHBlock);
    for (const auto & field_type : dag_context.result_field_types)
    {
        expected_types.emplace_back(getDataTypeByFieldTypeForComputingLayer(field_type));
    }
}

template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::finishWrite()
{
    assert(0 == rows_in_blocks);
    if (should_send_exec_summary_at_last)
        sendExecutionSummary();
}

template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::sendExecutionSummary()
{
    tipb::SelectResponse response;
    summary_collector.addExecuteSummaries(response);
    writer->sendExecutionSummary(response);
}

template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::flush()
{
    if (rows_in_blocks > 0)
        partitionAndEncodeThenWriteBlocks();
}

template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::write(const Block & block)
{
    RUNTIME_CHECK_MSG(
        block.columns() == dag_context.result_field_types.size(),
        "Output column size mismatch with field type size");
    size_t rows = block.rows();
    rows_in_blocks += rows;
    if (rows > 0)
    {
        blocks.push_back(block);
    }

    if (static_cast<Int64>(rows_in_blocks) > partition_batch_limit)
        partitionAndEncodeThenWriteBlocks();
}

extern void WriteColumnData(const IDataType & type, const ColumnPtr & column, WriteBuffer & ostr, size_t offset, size_t limit);

template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::partitionAndEncodeThenWriteBlocks()
{
    // return partitionAndEncodeThenWriteBlocksTest();

    if (blocks.empty())
        return;

    // Set mpp packet data version to `1`
    auto tracked_packets = HashBaseWriterHelper::createPackets(partition_num, HashPartitionWriterV1);

    // Do NOT enable data compression when using local tunnel
    for (size_t part_id = 0; part_id < partition_num; ++part_id)
    {
        auto mode = writer->isLocal(part_id) ? mpp::CompressionMode::NONE : compression_mode;
        tracked_packets[part_id]->getPacket().mutable_compression()->set_mode(mode);
    }

    // Sum of all approximate block data memory size
    size_t ori_block_mem_size = 0;

    {
        assert(rows_in_blocks > 0);

        HashBaseWriterHelper::materializeBlocks(blocks);

        // All blocks are same, use one block's meta info as header
        Block dest_block_header = blocks.back().cloneEmpty();

        std::vector<String> partition_key_containers(collators.size());
        std::vector<std::vector<MutableColumns>> dest_columns(partition_num);
        [[maybe_unused]] size_t total_rows = 0, encoded_rows = 0;

        while (!blocks.empty())
        {
            const auto & block = blocks.back();
            block.checkNumberOfRows();
            assertBlockSchema(expected_types, block, "HashPartitionWriterV1");

            ori_block_mem_size += ApproxBlockBytes(block);
            total_rows += block.rows();

            auto && dest_tbl_cols = HashBaseWriterHelper::createDestColumns(block, partition_num);
            HashBaseWriterHelper::scatterColumns(block, partition_num, collators, partition_key_containers, partition_col_ids, dest_tbl_cols);
            blocks.pop_back();

            for (size_t part_id = 0; part_id < partition_num; ++part_id)
            {
                auto & columns = dest_tbl_cols[part_id];
                dest_columns[part_id].emplace_back(std::move(columns));
            }
        }

        size_t header_size = GetExtraInfoSize(dest_block_header);

        for (size_t part_id = 0; part_id < partition_num; ++part_id)
        {
            auto & part_columns = dest_columns[part_id];
            size_t part_rows = std::accumulate(part_columns.begin(), part_columns.end(), 0, [](const auto & r, const auto & columns) { return r + columns.front()->size(); });

            if (!part_rows)
                continue;

            size_t part_column_bytes = std::accumulate(part_columns.begin(), part_columns.end(), 0, [](auto res, const auto & columns) {
                for (const auto & elem : columns)
                    res += elem->byteSize();
                return res + 8 /*partition rows*/;
            });

            // Each partition encode format:
            //  header meta(include all row count of the partition);
            //  repeated:
            //      row count;
            //      columns data;
            size_t init_size = part_column_bytes + header_size;

            // Reserve enough memory buffer size
            auto output_buffer = std::make_unique<WriteBufferFromOwnString>(init_size);
            std::unique_ptr<CompressedCHBlockChunkCodec::CompressedWriteBuffer> compress_codec{};
            WriteBuffer * ostr_ptr = output_buffer.get();

            // Init compression writer
            if (tracked_packets[part_id]->getPacket().compression().mode() != mpp::CompressionMode::NONE)
            {
                compress_codec = std::make_unique<CompressedCHBlockChunkCodec::CompressedWriteBuffer>(
                    *output_buffer,
                    CompressionSettings(ToInternalCompressionMethod(compression_mode)),
                    init_size);
                ostr_ptr = compress_codec.get();
            }

            // Encode header
            EncodeHeader(*ostr_ptr, dest_block_header, part_rows);

            for (auto && columns : part_columns)
            {
                size_t rows = columns.front()->size();
                if (!rows)
                    continue;

                // Encode row count for next columns
                writeVarUInt(rows, *ostr_ptr);
                encoded_rows += rows;

                // Encode columns data
                for (size_t col_index = 0; col_index < dest_block_header.columns(); ++col_index)
                {
                    auto && col_type_name = dest_block_header.getByPosition(col_index);
                    WriteColumnData(*col_type_name.type, std::move(columns[col_index]), *ostr_ptr, 0, 0);
                }

                columns.clear();
            }

            // Flush rest buffer
            if (compress_codec)
                compress_codec->next();

            tracked_packets[part_id]->getPacket().add_chunks(output_buffer->releaseStr());
        }
        assert(encoded_rows == total_rows);
        assert(blocks.empty());
        rows_in_blocks = 0;
    }

    writePackets(std::move(tracked_packets));

    GET_METRIC(tiflash_exchange_data_bytes, type_hash_original_all).Increment(ori_block_mem_size);
}


template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::partitionAndEncodeThenWriteBlocksTest()
{
    if (blocks.empty())
        return;

    // Set mpp packet data version to `1`
    auto tracked_packets = HashBaseWriterHelper::createPackets(partition_num, HashPartitionWriterV1);

    // Do NOT enable data compression when using local tunnel
    for (size_t part_id = 0; part_id < partition_num; ++part_id)
    {
        auto mode = writer->isLocal(part_id) ? mpp::CompressionMode::NONE : compression_mode;
        tracked_packets[part_id]->getPacket().mutable_compression()->set_mode(mode);
    }

    // Sum of all approximate block data memory size
    size_t ori_block_mem_size = 0;

    {
        assert(rows_in_blocks > 0);

        HashBaseWriterHelper::materializeBlocks(blocks);
        Block dest_block_header = blocks.back().cloneEmpty();

        std::vector<String> partition_key_containers(collators.size());
        std::vector<std::vector<MutableColumns>> dest_columns(partition_num);
        [[maybe_unused]] size_t total_rows = 0, encoded_rows = 0;

        while (!blocks.empty())
        {
            const auto & block = blocks.back();
            block.checkNumberOfRows();
            assertBlockSchema(expected_types, block, "HashPartitionWriterV1");

            ori_block_mem_size += ApproxBlockBytes(block);
            total_rows += block.rows();

            auto dest_tbl_cols = HashBaseWriterHelper::createDestColumns(block, partition_num);
            HashBaseWriterHelper::scatterColumns(block, partition_num, collators, partition_key_containers, partition_col_ids, dest_tbl_cols);
            blocks.pop_back();

            for (size_t part_id = 0; part_id < partition_num; ++part_id)
            {
                auto & columns = dest_tbl_cols[part_id];
                dest_columns[part_id].emplace_back(std::move(columns));
            }
        }

        size_t header_size = GetExtraInfoSize(dest_block_header) + 8;

        for (size_t part_id = 0; part_id < partition_num; ++part_id)
        {
            auto & part_columns = dest_columns[part_id];
            size_t part_rows = std::accumulate(part_columns.begin(), part_columns.end(), 0, [](const auto & r, const auto & columns) { return r + columns.front()->size(); });

            if (!part_rows)
                continue;

            encoded_rows += part_rows;

            size_t part_column_bytes = std::accumulate(
                part_columns.begin(),
                part_columns.end(),
                0,
                [](auto res, const auto & columns) {
                    for (const auto & elem : columns)
                        res += elem->byteSize();
                    return res + 8;
                });

            size_t init_size = part_column_bytes + header_size;

            // Reserve enough memory buffer size
            auto output_buffer = std::make_unique<WriteBufferFromOwnString>(init_size);
            std::unique_ptr<CompressedCHBlockChunkCodec::CompressedWriteBuffer> compress_codec{};
            WriteBuffer * ostr_ptr = output_buffer.get();

            // Init compression writer
            if (tracked_packets[part_id]->getPacket().compression().mode() != mpp::CompressionMode::NONE)
            {
                compress_codec = std::make_unique<CompressedCHBlockChunkCodec::CompressedWriteBuffer>(
                    *output_buffer,
                    CompressionSettings(ToInternalCompressionMethod(compression_mode)),
                    init_size);
                ostr_ptr = compress_codec.get();
            }

            // Encode header
            EncodeHeader(*ostr_ptr, dest_block_header, part_rows);
            writeVarUInt(part_columns.size(), *ostr_ptr);
            for (auto && columns : part_columns)
            {
                writeVarUInt(columns.front()->size(), *ostr_ptr);
            }

            for (size_t col_index = 0; col_index < dest_block_header.columns(); ++col_index)
            {
                auto && col_type_name = dest_block_header.getByPosition(col_index);
                for (auto && columns : part_columns)
                {
                    if (columns[col_index]->empty())
                        continue;
                    WriteColumnData(*col_type_name.type, std::move(columns[col_index]), *ostr_ptr, 0, 0);
                }
            }

            // Flush rest buffer
            if (compress_codec)
                compress_codec->next();

            tracked_packets[part_id]->getPacket().add_chunks(output_buffer->releaseStr());
        }

        assert(encoded_rows == total_rows);
        assert(blocks.empty());
        rows_in_blocks = 0;
    }

    writePackets(std::move(tracked_packets));

    GET_METRIC(tiflash_exchange_data_bytes, type_hash_original_all).Increment(ori_block_mem_size);
}

static void updateHashPartitionWriterMetrics(mpp::CompressionMode mode, size_t sz, bool is_local)
{
    switch (ToInternalCompressionMethod(mode))
    {
    case CompressionMethod::NONE:
    {
        if (is_local)
        {
            GET_METRIC(tiflash_exchange_data_bytes, type_hash_none_local).Increment(sz);
        }
        else
        {
            GET_METRIC(tiflash_exchange_data_bytes, type_hash_none).Increment(sz);
        }
        break;
    }
    case CompressionMethod::LZ4:
    {
        GET_METRIC(tiflash_exchange_data_bytes, type_hash_lz4).Increment(sz);
        break;
    }
    case CompressionMethod::ZSTD:
    {
        GET_METRIC(tiflash_exchange_data_bytes, type_hash_zstd).Increment(sz);
        break;
    }
    default:
        break;
    }
}

template <class ExchangeWriterPtr>
void HashPartitionWriterImplV1<ExchangeWriterPtr>::writePackets(TrackedMppDataPacketPtrs && packets)
{
    for (size_t part_id = 0; part_id < packets.size(); ++part_id)
    {
        auto & packet = packets[part_id];
        assert(packet);

        auto & inner_packet = packet->getPacket();

        if (auto sz = inner_packet.ByteSizeLong(); likely(inner_packet.chunks_size() > 0))
        {
            auto mode = inner_packet.compression().mode();
            writer->partitionWrite(std::move(packet), part_id);
            updateHashPartitionWriterMetrics(mode, sz, writer->isLocal(part_id));
        }
    }
}

template class HashPartitionWriterImplV1<MPPTunnelSetPtr>;

} // namespace DB