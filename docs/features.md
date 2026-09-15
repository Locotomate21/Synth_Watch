# Feature catalogue

Generated from the `FeatureSpec` declarations in `synthwatch.detect` by
`scripts/gen_feature_docs.py`. Do not edit by hand.

Every feature declares why it should carry signal and who it misclassifies
while behaving perfectly normally. The second half is the one that has to
reach the reader of a report.

## `coord_partner_count`

Number of distinct accounts this account co-published near-duplicates with.

- **Level**: account · **Unit**: count · **Direction**: higher = more anomalous
- **Rationale**: Operating several accounts from one content queue leaves the same text in several timelines at once, so partner count grows with the size of the queue rather than with the size of the audience.
- **Known limitation**: Anyone amplifying a widely circulated template -- a protest call, a fundraising appeal, a viral format -- accumulates partners without any coordination at all.

## `coord_pair_count`

Total near-duplicate co-posting events involving this account.

- **Level**: account · **Unit**: count · **Direction**: higher = more anomalous
- **Rationale**: Repetition separates a one-off coincidence from a sustained pattern: a shared template fires once, a shared queue fires repeatedly over the collection window.
- **Known limitation**: Scales with how much the account posts, so a prolific hobbyist outranks a quiet automated account; it must be read alongside the account volume features.

## `coord_max_similarity`

Highest fingerprint similarity reached with any co-publishing partner.

- **Level**: account · **Unit**: ratio · **Direction**: higher = more anomalous
- **Rationale**: Near-verbatim reuse is cheap to produce and hard to arrive at independently, so the ceiling of the similarity distribution is more telling than its mean.
- **Known limitation**: Quoting the same headline, press release or song lyric produces a perfect score, and short posts reach high similarity for purely combinatorial reasons.

## `coord_median_lag_seconds`

Median delay between this account and its partners on matched posts.

- **Level**: account · **Unit**: seconds · **Direction**: lower = more anomalous
- **Rationale**: Humans read, decide and type; dispatch systems do not. Median lags of a few seconds across many pairs indicate a shared trigger rather than a shared interest.
- **Known limitation**: Scheduling tools used openly by newsrooms and campaigns produce the same tight lags, and a push notification can synchronise thousands of genuine readers.

## `coord_duplicate_post_ratio`

Share of the account's eligible posts that matched someone else's.

- **Level**: account · **Unit**: ratio · **Direction**: higher = more anomalous
- **Rationale**: Normalises for volume: it separates an account whose entire output is recycled from one that occasionally shares a template alongside original writing.
- **Known limitation**: Accounts that exist to relay official statements -- mirrors, bulletins, aggregators -- legitimately sit near 1.0 and are not covert in any way.

## `coord_cluster_size`

Size of the Louvain community this account belongs to, else 1.

- **Level**: account · **Unit**: count · **Direction**: higher = more anomalous
- **Rationale**: Membership in a large, densely connected community is a property of the group rather than of the individual, which is the level this library reports at.
- **Known limitation**: Louvain is resolution-dependent and non-deterministic without a fixed seed; the same graph yields different community sizes under different resolutions.

## `coord_cluster_density`

Edge density of the account's community, 0 when it has none.

- **Level**: account · **Unit**: ratio · **Direction**: higher = more anomalous
- **Rationale**: A near-complete subgraph means every member matched every other member, which is far harder to reach by shared interest than a sparse chain of overlaps.
- **Known limitation**: Density falls mechanically as clusters grow, so it cannot be compared across clusters of very different sizes without normalisation.
