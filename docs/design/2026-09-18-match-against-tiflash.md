# TiFlash MATCH AGAINST pushdown

## Introduction

This document describes the TiFlash-side implementation for evaluating TiDB
`MATCH ... AGAINST` and `FTS_MATCH_WORD` expressions on the read snapshot.
The implementation is intentionally limited to the existing release-8.5
TiDB-to-TiFlash protocol. TiKV, TICI, and TiDB-side changes are out of scope.

## Motivation or Background

The TiDB planner can serialize full-text scalar signatures and the
`FTSQueryTypeWithScore` table-scan metadata into tipb, but the release-8.5
TiFlash branch did not recognize those signatures. A snapshot-local evaluator
allows TiFlash to execute the predicate and materialize `_FTS_SCORE` after
reading the row, without depending on an asynchronously maintained external
index.

## Detailed Design

### Protocol and planner path

The `contrib/tipb` submodule is advanced to the upstream protocol revision that
contains `ScalarFuncSig_FTSMatchWord`, `ScalarFuncSig_FTSMatchExpression`,
`FTSQueryInfo`, and `used_columnar_indexes`.

TiFlash maps the two scalar signatures as follows:

| tipb signature | TiFlash function | semantics |
| --- | --- | --- |
| `FTSMatchWord` | `fts_match_word` | Boolean-mode filter evaluator |
| `FTSMatchExpression` | `fts_match_expression` | Boolean-aware evaluator over MATCH columns |

For an index-style `FTSQueryInfo`, `PhysicalTableScan` reconstructs a
`fts_match_word(query, column)` filter and merges it with ordinary Selection
conditions. For `FTSQueryTypeWithScore`, it evaluates the same expression after
the scan and replaces the `_FTS_SCORE` generated-column placeholder with the
resulting `FLOAT` column. This keeps the existing table-scan schema and lets
Projection, Filter, and TopN consume the same score column. The expression
pipeline runs on the same read snapshot as the row data.

### Analyzer and matcher

The evaluator implements the STANDARD_V1-compatible Boolean matching subset
used by #70484/#70485:

- Unicode letter/number token runs with `_` preserved;
- Unicode lower-casing;
- default token length range 3..84;
- the default InnoDB stopword set;
- Boolean `+` required terms, `-` prohibited terms, quoted phrases, and a
  trailing `*` prefix;
- phrase positions are retained across analyzer filtering, so removed
  stopwords do not close phrase gaps;
- NULL MATCH columns contribute no tokens and do not nullify the whole row.

The predicate path consumes the numeric result as a Boolean value (`0` means
no match, a positive value means match). Queries containing Boolean syntax use
the Boolean matcher; plain queries retain the existing token-overlap behavior
for compatibility with the current scalar signature. The implementation
still returns a deterministic positive term-frequency value for the existing
`_FTS_SCORE` placeholder path, but that score is not part of the
#70484/#70485 acceptance scope and is not claimed to be MySQL/InnoDB
relevance. Nullable query arguments preserve NULL; nullable MATCH columns
contribute no tokens.

### Consistency boundary

This is a scan evaluator, not a DeltaMerge native full-text index. It therefore
does not introduce a second mutable index and cannot return rows that are
invisible at the read timestamp. The index id in `FTSQueryInfo` is used only to
identify the query contract in this phase; it is not read by a separate index
reader. The score is calculated from the query and row text, so it does not
provide corpus-wide IDF normalization.

## Test Design

### Functional Tests

`dbms/src/Functions/tests/gtest_fulltext.cpp` covers required/prohibited terms,
phrases with stopword gaps, prefixes, word boundaries, nullable input, NULL
MATCH columns, and unsupported score modifiers.

### Compatibility Tests

The normal local TiFlash read path and region-retry remote-read path use the
same expression evaluator. TICI/Disaggregated execution is intentionally not
part of this phase.

### Benchmark Tests

No benchmark is included in this phase. The scan evaluator is expected to be
slower than an inverted index and should be treated as compute pushdown, not
index-accelerated search.

## Impacts & Risks

The implementation adds per-row tokenization and matching CPU cost. Its score
is deterministic but is not a native MySQL relevance score. Custom analyzer
sysvars, custom stopword lists, NGRAM_V1, and non-default `AGAINST` modifiers
are not represented in the current wire contract and must not be treated as
supported by this phase.

## Investigation & Alternatives

An asynchronous external index was rejected because it cannot provide the
required snapshot consistency. A native DeltaMerge full-text index remains a
long-term performance design, but it requires a larger write-path, snapshot,
and index-reader change set.

## Unresolved Questions

If exact corpus-wide ranking is required, a future phase must define a
TiDB-compatible relevance-score contract and add a native DeltaMerge full-text
index.
