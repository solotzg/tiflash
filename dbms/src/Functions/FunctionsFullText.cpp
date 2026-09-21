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

#include <Columns/ColumnConst.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Common/StringUtils/StringUtils.h>
#include <Common/UTF8Helpers.h>
#include <Common/typeid_cast.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/FunctionsFullText.h>
#include <Poco/Unicode.h>
#include <Poco/UTF8String.h>
#include <tipb/executor.pb.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace DB
{
namespace ErrorCodes
{
extern const int ILLEGAL_COLUMN;
}

namespace
{
constexpr size_t default_min_token_size = 3;
constexpr size_t default_max_token_size = 84;
constexpr std::string_view fts_boolean_query_marker = "__tiflash_fts_boolean_query__:";

struct FullTextToken
{
    String text;
    size_t position = 0;
};

struct BooleanClause
{
    enum class Modifier
    {
        Should,
        Must,
        MustNot,
    };

    Modifier modifier = Modifier::Should;
    bool phrase = false;
    bool prefix = false;
    std::vector<String> terms;
    std::vector<size_t> offsets;
};

using FullTextColumn = std::vector<FullTextToken>;
using FullTextDocument = std::vector<FullTextColumn>;

bool isFullTextToken(UInt32 code_point)
{
    return isAlphaNumericASCII(static_cast<char>(code_point)) || code_point == '_' || Poco::Unicode::isAlpha(code_point)
        || Poco::Unicode::isDigit(code_point);
}

std::pair<UInt32, size_t> decodeCodePoint(std::string_view text, size_t offset)
{
    const auto decoded = UTF8::utf8Decode(text.data() + offset, text.size() - offset);
    if (decoded.second == 0 || decoded.first == UTF8::UTF8_Error || decoded.second > text.size() - offset)
        return {static_cast<UInt8>(text[offset]), 1};
    return {decoded.first, decoded.second};
}

std::vector<FullTextToken> tokenizeText(std::string_view text)
{
    std::vector<FullTextToken> result;
    size_t position = 0;
    for (size_t i = 0; i < text.size();)
    {
        const auto [code_point, length] = decodeCodePoint(text, i);
        if (!isFullTextToken(code_point))
        {
            i += length;
            continue;
        }

        String token;
        while (i < text.size())
        {
            const auto [code_point, length] = decodeCodePoint(text, i);
            if (!isFullTextToken(code_point))
                break;
            token.append(text.data() + i, length);
            i += length;
        }

        // The TiDB analyzer lower-cases after tokenization. Poco uses the
        // same Unicode tables for the C++ implementation, while also
        // preserving the ASCII fast path used by the old evaluator.
        token = Poco::UTF8::toLower(token);
        result.push_back({std::move(token), position++});
    }
    return result;
}

bool isDefaultStopword(const String & token)
{
    static const std::unordered_set<String> stopwords{
        "a",    "about", "an",   "are",  "as",   "at",   "be",   "by",   "com", "de", "en", "for",
        "from", "how",   "i",    "in",    "is",    "it",   "la",   "of",   "on",   "or",   "that", "the",
        "this", "to",    "was",  "what",  "when",  "where", "who",  "will", "with", "und", "www"};
    return stopwords.contains(token);
}

std::vector<FullTextToken> analyzeText(std::string_view text)
{
    std::vector<FullTextToken> result;
    for (auto & token : tokenizeText(text))
    {
        const auto code_points = UTF8::countCodePoints(
            reinterpret_cast<const UInt8 *>(token.text.data()),
            token.text.size());
        if (code_points >= default_min_token_size && code_points <= default_max_token_size
            && !isDefaultStopword(token.text))
            result.push_back(std::move(token));
    }
    return result;
}

bool isBooleanWhitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool parseBooleanQuery(std::string_view query, std::vector<BooleanClause> & clauses)
{
    for (size_t i = 0; i < query.size();)
    {
        while (i < query.size() && isBooleanWhitespace(query[i]))
            ++i;
        if (i == query.size())
            break;

        BooleanClause clause;
        // The STANDARD parser used by #70485 accepts only the operators that
        // affect filtering. Other InnoDB operators affect scoring or phrase
        // proximity and must not be silently treated as ordinary text.
        if (query[i] == '%' || query[i] == '(' || query[i] == ')' || query[i] == '<' || query[i] == '>'
            || query[i] == '~' || query[i] == '@')
            return false;

        if (query[i] == '+' || query[i] == '-')
        {
            clause.modifier = query[i] == '+' ? BooleanClause::Modifier::Must : BooleanClause::Modifier::MustNot;
            ++i;
        }

        if (i == query.size())
            return false;

        // InnoDB accepts a leading wildcard as a no-op. A trailing wildcard
        // is handled below and turns a term into a prefix query.
        if (query[i] == '*')
        {
            ++i;
            if (i == query.size())
                continue;
        }

        String raw;
        if (query[i] == '"')
        {
            clause.phrase = true;
            ++i;
            const size_t start = i;
            while (i < query.size() && query[i] != '"' && query[i] != '\n')
                ++i;
            if (i == query.size() || query[i] != '"')
                return false;
            raw.assign(query.substr(start, i - start));
            ++i;
        }
        else
        {
            const size_t start = i;
            while (i < query.size() && !isBooleanWhitespace(query[i]) && query[i] != '+' && query[i] != '-'
                   && query[i] != '*' && query[i] != '%' && query[i] != '(' && query[i] != ')' && query[i] != '<'
                   && query[i] != '>' && query[i] != '~' && query[i] != '@')
                ++i;
            if (start == i)
                return false;
            raw.assign(query.substr(start, i - start));
            if (!raw.empty() && raw.back() == '*')
            {
                clause.prefix = true;
                raw.pop_back();
            }
        }

        const auto terms = clause.prefix ? tokenizeText(raw) : analyzeText(raw);
        if (clause.prefix && terms.size() != 1)
            continue;

        const size_t first_position = terms.empty() ? 0 : terms.front().position;
        for (const auto & term : terms)
        {
            const auto code_points = UTF8::countCodePoints(
                reinterpret_cast<const UInt8 *>(term.text.data()),
                term.text.size());
            if (code_points <= default_max_token_size && (clause.prefix || code_points >= default_min_token_size)
                && (clause.prefix || !isDefaultStopword(term.text)))
            {
                clause.terms.push_back(term.text);
                if (clause.phrase)
                    clause.offsets.push_back(term.position - first_position);
            }
        }
        if (!clause.terms.empty() || clause.modifier == BooleanClause::Modifier::Must)
            clauses.push_back(std::move(clause));
    }
    return true;
}

bool matchesPhraseInColumn(const BooleanClause & clause, const FullTextColumn & document)
{
    if (clause.terms.empty() || clause.terms.size() != clause.offsets.size())
        return false;

    for (const auto & start : document)
    {
        bool matched = true;
        for (size_t i = 0; i < clause.terms.size(); ++i)
        {
            const auto expected_position = start.position + clause.offsets[i];
            const auto it = std::find_if(document.begin(), document.end(), [&](const FullTextToken & token) {
                return token.position == expected_position;
            });
            if (it == document.end() || it->text != clause.terms[i])
            {
                matched = false;
                break;
            }
        }
        if (matched)
            return true;
    }
    return false;
}

bool matchesClause(const BooleanClause & clause, const FullTextDocument & document)
{
    if (clause.terms.empty())
        return false;

    if (clause.phrase)
        return std::any_of(document.begin(), document.end(), [&](const FullTextColumn & column) {
            return matchesPhraseInColumn(clause, column);
        });

    return std::all_of(clause.terms.begin(), clause.terms.end(), [&](const String & term) {
        return std::any_of(document.begin(), document.end(), [&](const FullTextColumn & column) {
            return std::any_of(column.begin(), column.end(), [&](const FullTextToken & token) {
                return clause.prefix ? token.text.starts_with(term) : token.text == term;
            });
        });
    });
}

size_t countClauseMatches(const BooleanClause & clause, const FullTextDocument & document)
{
    if (!matchesClause(clause, document))
        return 0;

    if (!clause.phrase)
    {
        size_t count = 0;
        for (const auto & term : clause.terms)
        {
            for (const auto & column : document)
            {
                count += std::count_if(column.begin(), column.end(), [&](const FullTextToken & token) {
                    return clause.prefix ? token.text.starts_with(term) : token.text == term;
                });
            }
        }
        return count;
    }

    size_t count = 0;
    for (const auto & column : document)
    {
        for (const auto & start : column)
        {
            bool matched = true;
            for (size_t i = 0; i < clause.terms.size(); ++i)
            {
                const auto expected_position = start.position + clause.offsets[i];
                const auto it = std::find_if(column.begin(), column.end(), [&](const FullTextToken & token) {
                    return token.position == expected_position;
                });
                if (it == column.end() || it->text != clause.terms[i])
                {
                    matched = false;
                    break;
                }
            }
            if (matched)
                ++count;
        }
    }
    return count;
}

FullTextColumn analyzeColumn(std::string_view document)
{
    return analyzeText(document);
}

Float64 matchBooleanScore(const std::vector<BooleanClause> & clauses, const FullTextDocument & document)
{
	if (clauses.empty())
		return 0;

	bool has_positive = false;
	bool has_must = false;
	bool positive_match = false;
	Float64 score = 0;
	for (const auto & clause : clauses)
	{
		const bool matched = matchesClause(clause, document);
		const auto clause_score = static_cast<Float64>(countClauseMatches(clause, document));
		switch (clause.modifier)
		{
		case BooleanClause::Modifier::Must:
			has_must = true;
			if (!matched)
				return 0;
			score += clause_score;
			break;
		case BooleanClause::Modifier::MustNot:
			if (matched)
				return 0;
			break;
		case BooleanClause::Modifier::Should:
			has_positive = true;
			positive_match = positive_match || matched;
			score += clause_score;
			break;
		}
	}

	// With a required term, unprefixed terms are optional. Without one, at
	// least one unprefixed term must match, matching BOOLEAN MODE semantics.
	if (!(has_must || !has_positive || positive_match))
		return 0;
	// A query containing only prohibited terms has no positive term to score,
	// but an accepted row still has to pass the boolean filter.
	return score > 0 ? score : 1;
}

Float64 matchBooleanScore(std::string_view query, const FullTextDocument & document)
{
    std::vector<BooleanClause> clauses;
    if (!parseBooleanQuery(query, clauses))
        return 0;
    return matchBooleanScore(clauses, document);
}

Float64 matchBooleanScore(const tipb::FTSBooleanQuery & query, const FullTextDocument & document)
{
	std::vector<BooleanClause> clauses;
	for (const auto & node : query.nodes())
	{
		if (!node.has_term())
			return 0;

		const auto & term = node.term();
		BooleanClause clause;
		switch (node.occur())
		{
		case tipb::FTSBooleanOccur::FTSBooleanOccurMust:
			clause.modifier = BooleanClause::Modifier::Must;
			break;
		case tipb::FTSBooleanOccur::FTSBooleanOccurMustNot:
			clause.modifier = BooleanClause::Modifier::MustNot;
			break;
		case tipb::FTSBooleanOccur::FTSBooleanOccurShould:
			clause.modifier = BooleanClause::Modifier::Should;
			break;
		default:
			return 0;
		}

		switch (term.term_type())
		{
		case tipb::FTSBooleanTermType::FTSBooleanTermWord:
			for (const auto & token : analyzeText(term.text()))
				clause.terms.push_back(token.text);
			break;
		case tipb::FTSBooleanTermType::FTSBooleanTermPrefix:
		{
			clause.prefix = true;
			const auto terms = tokenizeText(term.text());
			if (terms.size() != 1)
			{
				if (clause.modifier == BooleanClause::Modifier::Must)
					clauses.push_back(std::move(clause));
				continue;
			}
			const auto code_points = UTF8::countCodePoints(
				reinterpret_cast<const UInt8 *>(terms.front().text.data()),
				terms.front().text.size());
			if (code_points > default_max_token_size)
			{
				if (clause.modifier == BooleanClause::Modifier::Must)
					clauses.push_back(std::move(clause));
				continue;
			}
			clause.terms.push_back(terms.front().text);
			break;
		}
		case tipb::FTSBooleanTermType::FTSBooleanTermPhrase:
		{
			clause.phrase = true;
			const auto terms = analyzeText(term.text());
			const size_t first_position = terms.empty() ? 0 : terms.front().position;
			for (const auto & token : terms)
			{
				clause.terms.push_back(token.text);
				clause.offsets.push_back(token.position - first_position);
			}
			break;
		}
		default:
			return 0;
		}

		if (!clause.terms.empty() || clause.modifier == BooleanClause::Modifier::Must)
			clauses.push_back(std::move(clause));
	}

	// A BOOLEAN MODE query containing only prohibited terms has no positive
	// branch. TiDB's local evaluator treats it as matching no rows.
	if (std::none_of(clauses.begin(), clauses.end(), [](const BooleanClause & clause) {
			return clause.modifier != BooleanClause::Modifier::MustNot;
		}))
		return 0;

	// The protocol path is the no-score MATCH ... AGAINST BOOLEAN MODE
	// predicate introduced by #70484/#70485. TiDB's local evaluator returns
	// a boolean 0/1 result, so do not expose term-frequency counts here.
	return matchBooleanScore(clauses, document) > 0 ? 1 : 0;
}

Float64 matchBooleanScore(std::string_view query, std::string_view document)
{
    return matchBooleanScore(query, FullTextDocument{analyzeColumn(document)});
}

size_t countTermMatches(const String & term, const FullTextColumn & document)
{
    return std::count_if(document.begin(), document.end(), [&](const FullTextToken & token) {
        return token.text == term;
    });
}

Float64 matchNaturalLanguageScore(std::string_view query, const FullTextDocument & document)
{
    const auto query_tokens = analyzeText(query);
    if (query_tokens.empty())
        return 0;

    std::unordered_set<String> unique_query_tokens;
    Float64 score = 0;
    for (const auto & query_token : query_tokens)
    {
        if (!unique_query_tokens.insert(query_token.text).second)
            continue;
        for (const auto & column : document)
            score += static_cast<Float64>(countTermMatches(query_token.text, column));
    }
    return score;
}

bool queryUsesBooleanSyntax(std::string_view query)
{
    return query.find_first_of("+-\"*%()<>~@") != std::string_view::npos;
}

String getStringAt(const IColumn & column, size_t row)
{
    if (const auto * column_nullable = typeid_cast<const ColumnNullable *>(&column))
    {
        if (column_nullable->getNullMapData()[row])
            return {};
        return getStringAt(column_nullable->getNestedColumn(), row);
    }

    if (const auto * column_const = typeid_cast<const ColumnConst *>(&column))
        return column_const->getValue<String>();

    const auto * column_string = checkAndGetColumn<ColumnString>(&column);
    if (column_string == nullptr)
        throw Exception("Full-text arguments must be string columns", ErrorCodes::ILLEGAL_COLUMN);

    const auto & chars = column_string->getChars();
    const auto & offsets = column_string->getOffsets();
    const size_t begin = row == 0 ? 0 : offsets[row - 1];
    const size_t end = offsets[row];
    return String(reinterpret_cast<const char *>(&chars[begin]), end - begin - 1);
}

bool isNullAt(const IColumn & column, size_t row)
{
    if (const auto * column_nullable = typeid_cast<const ColumnNullable *>(&column))
        return column_nullable->getNullMapData()[row];
    if (const auto * column_const = typeid_cast<const ColumnConst *>(&column))
    {
        if (const auto * nested_nullable = typeid_cast<const ColumnNullable *>(&column_const->getDataColumn()))
            return nested_nullable->getNullMapData()[0];
    }
    return false;
}

bool decodeFTSBooleanQuery(const IColumn & column, tipb::FTSBooleanQuery & query)
{
    const auto * constant = typeid_cast<const ColumnConst *>(&column);
    if (constant == nullptr)
        return false;
    const auto encoded = constant->getValue<String>();
    const std::string_view value(encoded);
    if (!value.starts_with(fts_boolean_query_marker))
        return false;
    return query.ParseFromString(std::string(value.substr(fts_boolean_query_marker.size())));
}

class FunctionFTSMatchWord final : public IFunction
{
public:
    static constexpr auto name = "fts_match_word";
    static FunctionPtr create(const Context &) { return std::make_shared<FunctionFTSMatchWord>(); }

    String getName() const override { return name; }
    size_t getNumberOfArguments() const override { return 2; }
    bool useDefaultImplementationForConstants() const override { return false; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {0}; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        bool nullable = false;
        for (const auto & argument : arguments)
        {
            if (!removeNullable(argument)->isString())
                throw Exception(
                    "Illegal type " + argument->getName() + " of argument of function " + getName(),
                    ErrorCodes::ILLEGAL_COLUMN);
            nullable = nullable || argument->isNullable();
        }
        DataTypePtr result_type = std::make_shared<DataTypeFloat64>();
        return nullable ? std::make_shared<DataTypeNullable>(result_type) : result_type;
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result) const override
    {
        const auto * query_column = typeid_cast<const ColumnConst *>(&*block.getByPosition(arguments[0]).column);
        if (query_column == nullptr)
            throw Exception(
                "The query argument of fts_match_word must be constant",
                ErrorCodes::ILLEGAL_COLUMN);

        const auto & document_column = block.getByPosition(arguments[1]).column;
        auto output = ColumnFloat64::create(document_column->size());
        auto & output_data = output->getData();

        if (isNullAt(*query_column, 0))
        {
            auto null_map = ColumnUInt8::create(document_column->size(), 1);
            block.getByPosition(result).column = ColumnNullable::create(std::move(output), std::move(null_map));
            return;
        }
        const auto query = query_column->getValue<String>();

        if (block.getByPosition(arguments[1]).type->isNullable() || document_column->isColumnNullable())
        {
            auto null_map = ColumnUInt8::create(document_column->size(), 0);
            auto & null_map_data = null_map->getData();
            for (size_t row = 0; row < document_column->size(); ++row)
            {
                if (isNullAt(*document_column, row))
                    null_map_data[row] = 1;
                else
                    output_data[row] = matchBooleanScore(query, getStringAt(*document_column, row));
            }
            block.getByPosition(result).column = ColumnNullable::create(std::move(output), std::move(null_map));
            return;
        }

        if (const auto * document = checkAndGetColumn<ColumnString>(&*document_column))
        {
            const auto & chars = document->getChars();
            const auto & offsets = document->getOffsets();
            for (size_t row = 0, begin = 0; row < offsets.size(); ++row)
            {
                const size_t end = offsets[row];
                const size_t length = end - begin - 1;
                output_data[row]
                    = matchBooleanScore(query, std::string_view(reinterpret_cast<const char *>(&chars[begin]), length));
                begin = end;
            }
        }
        else if (const auto * document = typeid_cast<const ColumnConst *>(&*document_column))
        {
            const auto value = document->getValue<String>();
            const Float64 matched = matchBooleanScore(query, value);
            std::fill(output_data.begin(), output_data.end(), matched);
        }
        else
        {
            throw Exception(
                "Illegal column " + document_column->getName() + " of argument of function " + getName(),
                ErrorCodes::ILLEGAL_COLUMN);
        }
        block.getByPosition(result).column = std::move(output);
    }
};

class FunctionFTSMatchExpression final : public IFunction
{
public:
    static constexpr auto name = "fts_match_expression";
    static FunctionPtr create(const Context &) { return std::make_shared<FunctionFTSMatchExpression>(); }

    String getName() const override { return name; }
    size_t getNumberOfArguments() const override { return 0; }
    bool isVariadic() const override { return true; }
    bool useDefaultImplementationForConstants() const override { return false; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {0}; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() < 2)
            throw Exception("fts_match_expression requires a query and at least one column");
        bool nullable = false;
        for (const auto & argument : arguments)
        {
            if (!removeNullable(argument)->isString())
                throw Exception(
                    "Illegal type " + argument->getName() + " of argument of function " + getName(),
                    ErrorCodes::ILLEGAL_COLUMN);
            nullable = nullable || argument->isNullable();
        }
        DataTypePtr result_type = std::make_shared<DataTypeFloat64>();
        return nullable ? std::make_shared<DataTypeNullable>(result_type) : result_type;
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result) const override
    {
        const auto * query_column = typeid_cast<const ColumnConst *>(&*block.getByPosition(arguments[0]).column);
        if (query_column == nullptr)
            throw Exception(
                "The query argument of fts_match_expression must be constant",
                ErrorCodes::ILLEGAL_COLUMN);

        const size_t rows = block.getByPosition(arguments[1]).column->size();
        auto output = ColumnFloat64::create(rows, 0);
        auto & output_data = output->getData();
        const bool nullable = block.getByPosition(result).type->isNullable();
        auto null_map = nullable ? ColumnUInt8::create(rows, 0) : nullptr;
        auto * null_map_data = null_map ? &null_map->getData() : nullptr;

        if (isNullAt(*query_column, 0))
        {
            if (null_map_data != nullptr)
                std::fill(null_map_data->begin(), null_map_data->end(), 1);
            ColumnPtr result_column;
            if (null_map)
                result_column = ColumnNullable::create(std::move(output), std::move(null_map));
            else
                result_column = std::move(output);
            block.getByPosition(result).column = std::move(result_column);
            return;
        }
        const auto query = query_column->getValue<String>();
        size_t document_argument_end = arguments.size();
        tipb::FTSBooleanQuery protocol_boolean_query;
        const bool has_protocol_boolean_query = arguments.size() > 2
            && decodeFTSBooleanQuery(*block.getByPosition(arguments.back()).column, protocol_boolean_query);
        if (has_protocol_boolean_query)
            --document_argument_end;

        for (size_t row = 0; row < rows; ++row)
        {
            FullTextDocument document;
            document.reserve(document_argument_end - 1);
            for (size_t arg = 1; arg < document_argument_end; ++arg)
            {
                // A NULL MATCH column contributes no tokens. This is
                // intentional: #70485 relies on a row with a NULL body
                // still matching a required term from another MATCH column.
                if (isNullAt(*block.getByPosition(arguments[arg]).column, row))
                {
                    document.emplace_back();
                    continue;
                }
                document.push_back(analyzeColumn(getStringAt(*block.getByPosition(arguments[arg]).column, row)));
            }
            if (has_protocol_boolean_query)
                output_data[row] = matchBooleanScore(protocol_boolean_query, document);
            else if (queryUsesBooleanSyntax(query))
                output_data[row] = matchBooleanScore(query, document);
            else
                output_data[row] = matchNaturalLanguageScore(query, document);
        }
        ColumnPtr result_column;
        if (null_map)
            result_column = ColumnNullable::create(std::move(output), std::move(null_map));
        else
            result_column = std::move(output);
        block.getByPosition(result).column = std::move(result_column);
    }
};
}

void registerFunctionsFullText(FunctionFactory & factory)
{
    factory.registerFunction<FunctionFTSMatchWord>(FunctionFactory::CaseInsensitive);
    factory.registerFunction<FunctionFTSMatchExpression>(FunctionFactory::CaseInsensitive);
}
} // namespace DB
