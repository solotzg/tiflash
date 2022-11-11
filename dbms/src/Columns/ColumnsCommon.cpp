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

#include <Columns/ColumnsCommon.h>
#include <Columns/IColumn.h>
#include <common/memcpy.h>

#ifdef TIFLASH_ENABLE_AVX_SUPPORT
ASSERT_USE_AVX2_COMPILE_FLAG
#endif

namespace DB
{
#if defined(__AVX2__)
inline uint64_t ToBits64(const Int8 * bytes64)
{
    const auto check_block = _mm256_setzero_si256();
    uint64_t mask0 = mem_utils::details::get_block32_cmp_eq_mask(bytes64, check_block);
    uint64_t mask1 = mem_utils::details::get_block32_cmp_eq_mask(bytes64 + mem_utils::details::BLOCK32_SIZE, check_block);
    auto res = mask0 | (mask1 << mem_utils::details::BLOCK32_SIZE);
    return ~res;
}
#elif defined(__SSE2__)
/// Transform 64-byte mask to 64-bit mask.
inline UInt64 ToBits64(const Int8 * bytes64)
{
    const __m128i zero16 = _mm_setzero_si128();
    UInt64 res
        = static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64)), zero16)))
        | (static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 16)), zero16))) << 16)
        | (static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 32)), zero16))) << 32)
        | (static_cast<UInt64>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 48)), zero16))) << 48);

    return ~res;
}
#endif
#if defined(__SSE2__)
inline uint64_t ToBits16(const int8_t * bytes64)
{
    const auto check_block = _mm_setzero_si128();
    uint64_t res = mem_utils::details::get_block16_cmp_eq_mask(bytes64, check_block);
    return ~res;
}
#endif

static inline size_t countBytesInFilter(const UInt8 * filt, size_t start, size_t end)
{
#if defined(__AVX2__)
    size_t size = end - start;
    auto zero_cnt = mem_utils::details::avx2_byte_count(reinterpret_cast<const char *>(filt + start), size, 0);
    return size - zero_cnt;
#else
    size_t count = 0;

    /** NOTE: In theory, `filt` should only contain zeros and ones.
      * But, just in case, here the condition > 0 (to signed bytes) is used.
      * It would be better to use != 0, then this does not allow SSE2.
      */

    const char * pos = reinterpret_cast<const char *>(filt);
    pos += start;

    const char * end_pos = pos + (end - start);
    for (; pos < end_pos; ++pos)
        count += *pos != 0;

    return count;
#endif
}

size_t countBytesInFilter(const UInt8 * filt, size_t size)
{
    return countBytesInFilter(filt, 0, size);
}

size_t countBytesInFilter(const IColumn::Filter & filt)
{
    return countBytesInFilter(filt.data(), 0, filt.size());
}

static inline size_t _count_bytes_in_filter_with_null(const Int8 * p1, const Int8 * p2, size_t size)
{
    size_t count = 0;
    for (size_t i = 0; i < size; ++i)
    {
        count += (p1[i] & ~p2[i]) != 0;
    }
    return count;
}

static inline size_t countBytesInFilterWithNull(const IColumn::Filter & filt, const UInt8 * null_map, size_t start, size_t end)
{
    size_t count = 0;

    /** NOTE: In theory, `filt` should only contain zeros and ones.
      * But, just in case, here the condition > 0 (to signed bytes) is used.
      * It would be better to use != 0, then this does not allow SSE2.
      */

    const Int8 * pos = reinterpret_cast<const Int8 *>(filt.data()) + start;
    const Int8 * pos2 = reinterpret_cast<const Int8 *>(null_map) + start;

#if defined(__SSE2__) || defined(__AVX2__)
    size_t size = end - start;
    for (; size >= 64;)
    {
        count += __builtin_popcountll(ToBits64(pos) & ~ToBits64(pos2));
        pos += 64, pos2 += 64;
        size -= 64;
    }
#endif
    count += _count_bytes_in_filter_with_null(pos, pos2, size);
    return count;
}

size_t countBytesInFilterWithNull(const IColumn::Filter & filt, const UInt8 * null_map)
{
    return countBytesInFilterWithNull(filt, null_map, 0, filt.size());
}

std::vector<size_t> countColumnsSizeInSelector(IColumn::ColumnIndex num_columns, const IColumn::Selector & selector)
{
    std::vector<size_t> counts(num_columns);
    for (auto idx : selector)
        ++counts[idx];

    return counts;
}

namespace ErrorCodes
{
extern const int SIZES_OF_COLUMNS_DOESNT_MATCH;
}

namespace
{
/// Implementation details of filterArraysImpl function, used as template parameter.
/// Allow to build or not to build offsets array.

struct ResultOffsetsBuilder
{
    IColumn::Offsets & res_offsets;
    IColumn::Offset current_src_offset = 0;

    explicit ResultOffsetsBuilder(IColumn::Offsets * res_offsets_)
        : res_offsets(*res_offsets_)
    {}

    void reserve(ssize_t result_size_hint, size_t src_size)
    {
        res_offsets.reserve(result_size_hint > 0 ? result_size_hint : src_size);
    }

    void insertOne(size_t array_size)
    {
        current_src_offset += array_size;
        res_offsets.push_back(current_src_offset);
    }

    template <size_t SIMD_BYTES>
    void insertChunk(
        const IColumn::Offset * src_offsets_pos,
        bool first,
        IColumn::Offset chunk_offset,
        size_t chunk_size)
    {
        const auto offsets_size_old = res_offsets.size();
        res_offsets.resize(offsets_size_old + SIMD_BYTES);
        inline_memcpy(&res_offsets[offsets_size_old], src_offsets_pos, SIMD_BYTES * sizeof(IColumn::Offset));

        if (!first)
        {
            /// difference between current and actual offset
            const auto diff_offset = chunk_offset - current_src_offset;

            if (diff_offset > 0)
            {
                auto * res_offsets_pos = &res_offsets[offsets_size_old];

                /// adjust offsets
                for (size_t i = 0; i < SIMD_BYTES; ++i)
                    res_offsets_pos[i] -= diff_offset;
            }
        }
        current_src_offset += chunk_size;
    }
};

struct NoResultOffsetsBuilder
{
    explicit NoResultOffsetsBuilder(IColumn::Offsets *) {}
    void reserve(ssize_t, size_t) {}
    void insertOne(size_t) {}

    template <size_t SIMD_BYTES>
    void insertChunk(
        const IColumn::Offset *,
        bool,
        IColumn::Offset,
        size_t)
    {
    }
};


/// Transform 64-byte mask to 64-bit mask
inline uint64_t Bytes64MaskToBits64Mask(const UInt8 * bytes64)
{
#if defined(__AVX2__)
    const __m256i zero32 = _mm256_setzero_si256();
    uint64_t res = (static_cast<uint64_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bytes64)),
                        zero32)))
                    & 0xffffffff)
        | (static_cast<uint64_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
               _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bytes64 + 32)),
               zero32)))
           << 32);
#elif defined(__SSE2__)
    const __m128i zero16 = _mm_setzero_si128();
    uint64_t res = (static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64)),
                        zero16)))
                    & 0xffff)
        | ((static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 16)),
                zero16)))
            << 16)
           & 0xffff0000)
        | ((static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 32)),
                zero16)))
            << 32)
           & 0xffff00000000)
        | ((static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes64 + 48)),
                zero16)))
            << 48)
           & 0xffff000000000000);
#elif defined(__aarch64__) && defined(__ARM_NEON)
    const uint8x16_t bitmask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    const auto * src = reinterpret_cast<const unsigned char *>(bytes64);
    const uint8x16_t p0 = vceqzq_u8(vld1q_u8(src));
    const uint8x16_t p1 = vceqzq_u8(vld1q_u8(src + 16));
    const uint8x16_t p2 = vceqzq_u8(vld1q_u8(src + 32));
    const uint8x16_t p3 = vceqzq_u8(vld1q_u8(src + 48));
    uint8x16_t t0 = vandq_u8(p0, bitmask);
    uint8x16_t t1 = vandq_u8(p1, bitmask);
    uint8x16_t t2 = vandq_u8(p2, bitmask);
    uint8x16_t t3 = vandq_u8(p3, bitmask);
    uint8x16_t sum0 = vpaddq_u8(t0, t1);
    uint8x16_t sum1 = vpaddq_u8(t2, t3);
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    uint64_t res = vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
#else
    uint64_t res = 0;
    for (size_t i = 0; i < 64; ++i)
        res |= static_cast<uint64_t>(0 == bytes64[i]) << i;
#endif
    return ~res;
}

template <typename T, typename ResultOffsetsBuilder>
void filterArraysImplGeneric(
    const PaddedPODArray<T> & src_elems,
    const IColumn::Offsets & src_offsets,
    PaddedPODArray<T> & res_elems,
    IColumn::Offsets * res_offsets,
    const IColumn::Filter & filt,
    ssize_t result_size_hint)
{
    const size_t size = src_offsets.size();
    if (size != filt.size())
        throw Exception(fmt::format("size of filter {} doesn't match size of column {}", filt.size(), size), ErrorCodes::SIZES_OF_COLUMNS_DOESNT_MATCH);

    ResultOffsetsBuilder result_offsets_builder(res_offsets);

    if (result_size_hint)
    {
        result_offsets_builder.reserve(result_size_hint, size);

        if (result_size_hint < 0)
            res_elems.reserve(src_elems.size());
        else if (result_size_hint < 1000000000 && src_elems.size() < 1000000000) /// Avoid overflow.
            res_elems.reserve((result_size_hint * src_elems.size() + size - 1) / size);
    }

    const UInt8 * filt_pos = filt.data();
    const auto * filt_end = filt_pos + size;

    const auto * offsets_pos = src_offsets.data();
    const auto * offsets_begin = offsets_pos;

    /// copy array ending at *end_offset_ptr
    const auto copy_array = [&](const IColumn::Offset * offset_ptr) {
        const auto arr_offset = offset_ptr == offsets_begin ? 0 : offset_ptr[-1];
        const auto arr_size = *offset_ptr - arr_offset;

        result_offsets_builder.insertOne(arr_size);

        const auto elems_size_old = res_elems.size();
        res_elems.resize(elems_size_old + arr_size);
        inline_memcpy(&res_elems[elems_size_old], &src_elems[arr_offset], arr_size * sizeof(T));
    };

    /** A slightly more optimized version.
        * Based on the assumption that often pieces of consecutive values
        *  completely pass or do not pass the filter.
        * Therefore, we will optimistically check the parts of `SIMD_BYTES` values.
        */
    static constexpr size_t SIMD_BYTES = 64;
    const auto * filt_end_aligned = filt_pos + size / SIMD_BYTES * SIMD_BYTES;

    while (filt_pos < filt_end_aligned)
    {
        uint64_t mask = Bytes64MaskToBits64Mask(filt_pos);

        if (0xffffffffffffffff == mask)
        {
            /// SIMD_BYTES consecutive rows pass the filter
            const auto first = offsets_pos == offsets_begin;

            const auto chunk_offset = first ? 0 : offsets_pos[-1];
            const auto chunk_size = offsets_pos[SIMD_BYTES - 1] - chunk_offset;

            result_offsets_builder.template insertChunk<SIMD_BYTES>(offsets_pos, first, chunk_offset, chunk_size);

            /// copy elements for SIMD_BYTES arrays at once
            const auto elems_size_old = res_elems.size();
            res_elems.resize(elems_size_old + chunk_size);
            inline_memcpy(&res_elems[elems_size_old], &src_elems[chunk_offset], chunk_size * sizeof(T));
        }
        else
        {
            while (mask)
            {
                size_t index = __builtin_ctzll(mask);
                copy_array(offsets_pos + index);
                mask = mask & (mask - 1);
            }
        }

        filt_pos += SIMD_BYTES;
        offsets_pos += SIMD_BYTES;
    }

    while (filt_pos < filt_end)
    {
        if (*filt_pos)
            copy_array(offsets_pos);

        ++filt_pos;
        ++offsets_pos;
    }
}
} // namespace


template <typename T>
void filterArraysImpl(
    const PaddedPODArray<T> & src_elems,
    const IColumn::Offsets & src_offsets,
    PaddedPODArray<T> & res_elems,
    IColumn::Offsets & res_offsets,
    const IColumn::Filter & filt,
    ssize_t result_size_hint)
{
    return filterArraysImplGeneric<T, ResultOffsetsBuilder>(src_elems, src_offsets, res_elems, &res_offsets, filt, result_size_hint);
}

template <typename T>
void filterArraysImplOnlyData(
    const PaddedPODArray<T> & src_elems,
    const IColumn::Offsets & src_offsets,
    PaddedPODArray<T> & res_elems,
    const IColumn::Filter & filt,
    ssize_t result_size_hint)
{
    return filterArraysImplGeneric<T, NoResultOffsetsBuilder>(src_elems, src_offsets, res_elems, nullptr, filt, result_size_hint);
}


/// Explicit instantiations - not to place the implementation of the function above in the header file.
#define INSTANTIATE(TYPE)                         \
    template void filterArraysImpl<TYPE>(         \
        const PaddedPODArray<TYPE> &,             \
        const IColumn::Offsets &,                 \
        PaddedPODArray<TYPE> &,                   \
        IColumn::Offsets &,                       \
        const IColumn::Filter &,                  \
        ssize_t);                                 \
    template void filterArraysImplOnlyData<TYPE>( \
        const PaddedPODArray<TYPE> &,             \
        const IColumn::Offsets &,                 \
        PaddedPODArray<TYPE> &,                   \
        const IColumn::Filter &,                  \
        ssize_t);

INSTANTIATE(UInt8)
INSTANTIATE(UInt16)
INSTANTIATE(UInt32)
INSTANTIATE(UInt64)
INSTANTIATE(Int8)
INSTANTIATE(Int16)
INSTANTIATE(Int32)
INSTANTIATE(Int64)
INSTANTIATE(Float32)
INSTANTIATE(Float64)

#undef INSTANTIATE

} // namespace DB
