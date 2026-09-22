// Copyright 2026 PingCAP, Inc.
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

#include <TestUtils/FunctionTestUtils.h>
#include <TestUtils/TiFlashTestBasic.h>
#include <TiDB/Collation/Collator.h>
#include <gtest/gtest.h>
#include <tipb/executor.pb.h>

namespace DB::tests
{
class TestFullText : public DB::tests::FunctionTest
{
};

TEST_F(TestFullText, MatchWordBoolean)
try
{
    const auto query = createConstColumn<String>(4, "+quick -slow");
    const auto documents = createColumn<String>({
        "A quick brown fox",
        "A slow brown fox",
        "A brown fox",
        "A QUICK runner",
    });
    ASSERT_COLUMN_EQ(createColumn<Float64>({1, 0, 0, 1}), executeFunction("fts_match_word", {query, documents}));
}
CATCH

TEST_F(TestFullText, MatchWordPhraseAndPrefix)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 0}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(2, "\"quick brown\""), createColumn<String>({"quick brown fox", "quick fox brown"})}));
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 0}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(2, "run*"), createColumn<String>({"runner", "walk"})}));
}
CATCH

TEST_F(TestFullText, MatchWordTokenBoundaryAndPhrasePositions)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({0, 1}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(2, "cat"), createColumn<String>({"concatenate", "a cat"})}));
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({0, 1}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(2, "\"quick the brown\""),
             createColumn<String>({"quick brown", "quick the brown"})}));
}
CATCH

TEST_F(TestFullText, MatchWordNull)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Nullable<Float64>>({1, {}}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(2, "hello"), createColumn<Nullable<String>>({"hello world", {}})}));
}
CATCH

TEST_F(TestFullText, MatchWordScore)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 2, 0}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(3, "quick"), createColumn<String>({"quick", "quick quick", "slow"})}));
}
CATCH

TEST_F(TestFullText, MatchExpression)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 0}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(2, "quick fox"), createColumn<String>({"quick brown", "slow turtle"})}));
}
CATCH

TEST_F(TestFullText, MatchExpressionProtocolBooleanQuery)
try
{
    tipb::FTSBooleanQuery boolean_query;
    auto * required = boolean_query.add_nodes();
    required->set_occur(tipb::FTSBooleanOccurMust);
    required->mutable_term()->set_term_type(tipb::FTSBooleanTermWord);
    required->mutable_term()->set_text("quick");
    auto * prohibited = boolean_query.add_nodes();
    prohibited->set_occur(tipb::FTSBooleanOccurMustNot);
    prohibited->mutable_term()->set_term_type(tipb::FTSBooleanTermWord);
    prohibited->mutable_term()->set_text("slow");

    const String metadata = "__tiflash_fts_boolean_query__:" + boolean_query.SerializeAsString();
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 0, 0}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(3, "+quick -slow"),
             createColumn<String>({"quick brown", "slow fox", "brown fox"}),
             createConstColumn<String>(3, metadata)}));

    tipb::FTSBooleanQuery prohibited_query;
    auto * prohibited_only = prohibited_query.add_nodes();
    prohibited_only->set_occur(tipb::FTSBooleanOccurMustNot);
    prohibited_only->mutable_term()->set_term_type(tipb::FTSBooleanTermWord);
    prohibited_only->mutable_term()->set_text("slow");
    const String prohibited_metadata = "__tiflash_fts_boolean_query__:" + prohibited_query.SerializeAsString();
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({0, 0, 0}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(3, "-slow"),
             createColumn<String>({"quick brown", "slow fox", "brown fox"}),
             createConstColumn<String>(3, prohibited_metadata)}));
}
CATCH

TEST_F(TestFullText, MatchExpressionScore)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 3, 0}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(3, "quick fox"),
             createColumn<String>({"quick brown", "quick fox fox", "slow turtle"})}));
}
CATCH

TEST_F(TestFullText, MatchExpressionCollation)
try
{
    const auto ci_collator = TiDB::ITiDBCollator::getCollator(TiDB::ITiDBCollator::UTF8MB4_GENERAL_CI);
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 1}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(2, "+quick"), createColumn<String>({"QUICK runner", "quick runner"})},
            ci_collator));

    const auto binary_collator = TiDB::ITiDBCollator::getCollator(TiDB::ITiDBCollator::UTF8MB4_BIN);
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({0, 1}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(2, "+quick"), createColumn<String>({"QUICK runner", "quick runner"})},
            binary_collator));

    ASSERT_COLUMN_EQ(
        createColumn<Float64>({1, 0}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(2, "+run*"), createColumn<String>({"RUNNER", "walk"})},
            ci_collator));
}
CATCH

TEST_F(TestFullText, MatchExpressionNullColumn)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Nullable<Float64>>({0, 1}),
        executeFunction(
            "fts_match_expression",
            {createConstColumn<String>(2, "+indexing -postgresql"),
             createColumn<String>({"Indexing", "Indexing"}),
             createColumn<Nullable<String>>({"PostgreSQL", {}})}));
}
CATCH

TEST_F(TestFullText, MatchBooleanUnsupportedScoreOperator)
try
{
    ASSERT_COLUMN_EQ(
        createColumn<Float64>({0}),
        executeFunction(
            "fts_match_word",
            {createConstColumn<String>(1, ">quick"), createColumn<String>({"quick"})}));
}
CATCH
} // namespace DB::tests
