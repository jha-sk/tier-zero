# Attribution

`serverfault.json` is derived from the Server Fault Stack Exchange data dump
(2024-04-07), obtained from <https://archive.org/details/stackexchange>.

## What it contains

4,437 query/document pairs. Each carries:

- `query_title`, `query_body` — a question title and a 500-character body
  excerpt, **authored by Server Fault contributors**
- `gold_title`, `gold_tags`, `gold_score` — the canonical question this one was
  closed as a duplicate of, its tags, and its community score
- `query_id`, `gold_doc_id` — the original post ids

The relevance labels come from moderator duplicate judgements (`PostLinks` rows
with `LinkTypeId=3`): a moderator read two questions and ruled that one is
answered by the other. Those judgements are the reason this eval set exists —
they are relevance labels produced by domain practitioners rather than generated
by the class of model being evaluated.

## Licence

The question and answer content is licensed **CC BY-SA 4.0** by its original
authors, and remains under that licence here. It is *not* covered by the
repository's MIT licence.

See <https://stackoverflow.com/help/licensing> for Stack Exchange's terms.

Every pair resolves to its source: `https://serverfault.com/q/<query_id>` and
`https://serverfault.com/q/<gold_doc_id>`.

## What is not redistributed

The corpus itself — `Posts.xml`, 1.21 GB — is **not** in this repository. It is
downloaded at build time from archive.org using the command in the top-level
README, and the golden set is pinned to it by a content hash
(`corpus_hash` in the JSON) so that a recall figure computed against a different
corpus is distinguishable from one computed against this one.

## Regenerating

```bash
cargo run --release -p tz-eval --bin tz-golden -- data/serverfault
```

The code that produces this file is MIT, like the rest of the repository.
