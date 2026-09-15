# Internal schema

Every adapter normalises into three record types before any feature is
computed. The rules live in the module docstring of `synthwatch/models.py`;
this page covers the decisions worth arguing about.

## `Account`

A snapshot at collection time, not a time series. Follower counts are the
values seen when the profile was crawled, which may be long after the posts in
the same corpus were written — `collected_at` exists so that gap stays
auditable instead of invisible.

`declared_automated` (Mastodon's `bot` flag, a self-identifying bio) is stored
but is a **label source, never a feature**: training on it only teaches the
model to find the honest bots.

There is no `is_bot` field, and there will not be one.

## `Post`

`text` is stored verbatim — no case folding, no whitespace collapsing, no
Unicode normalisation — because typo rate, sentence-length uniformity and
SimHash all read the original characters. Normalisation happens per feature and
is documented with that feature.

`root_post_id` keeps one busy thread from being read as coordination between
the people replying inside it.

Platform vocabulary that does not map goes into `extra` rather than being
dropped, so a field that turns out to matter later can be recovered without
re-collecting the data.

## `LabelRecord`

Ground truth is attached out of band, never written onto the account, and
always with provenance: `dataset` and `method` are required because
suspension-derived and hand-annotated labels disagree systematically, and an
evaluation that mixes them without saying so is not reproducible.

## `Corpus`

Posts are sorted by `(created_at, post_id)` at construction. That gives the
sliding windows in `detect.coordination` one canonical ordering and makes tie
behaviour deterministic across runs and platforms.

Referential integrity is **reported, not enforced**: real collections contain
posts whose author was suspended before the crawl reached them.
`orphan_post_ids` and `silent_account_ids` surface it; `build_corpus(...,
strict=True)` refuses it.

## Time

Naive datetimes are rejected at validation. Circadian distribution and
inter-arrival entropy are meaningless if the offset was guessed, and a silent
"assume UTC" is exactly the kind of bug that yields a confident wrong answer.
