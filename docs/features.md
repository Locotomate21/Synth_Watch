# Feature catalogue

Generated from the `FeatureSpec` declarations in `synthwatch.detect` by
`scripts/gen_feature_docs.py`. Do not edit by hand.

Every feature declares why it should carry signal and who it misclassifies
while behaving perfectly normally. The second half is the one that has to
reach the reader of a report.

## `temp_circadian_entropy`

Evenness of posting across the 24 hours of the day.

- **Level**: account · **Unit**: bits_normalised · **Direction**: higher = more anomalous
- **Rationale**: People are awake on a schedule and their posting inherits the shape of it. A process that runs continuously spreads evenly across all 24 hours, which pushes this towards its maximum. The measure is invariant to timezone.
- **Known limitation**: Shift workers, insomniacs and accounts shared across several timezones all flatten the same way, and any account with few posts looks flat by accident.

## `temp_quiet_hours_share`

Share of posts inside the account's own quietest six-hour window.

- **Level**: account · **Unit**: ratio · **Direction**: higher = more anomalous
- **Rationale**: Everyone sleeps somewhere, so a human timeline has a stretch that is nearly empty wherever their night happens to fall. Finding the quietest window per account rather than assuming a local night keeps this usable without knowing anyone's timezone.
- **Known limitation**: An account operated in shifts by several people, or one that mostly reshares automatically while its owner sleeps, has no quiet window and is not covert.

## `temp_interarrival_entropy`

Entropy of the log-binned gaps between consecutive posts.

- **Level**: account · **Unit**: bits_normalised · **Direction**: lower = more anomalous
- **Rationale**: Human attention is lumpy: minutes of replies, then hours of nothing, spread over many orders of magnitude. A fixed cadence collapses every gap into one bucket, and the entropy falls towards zero.
- **Known limitation**: Any scheduling tool produces the same collapse, and an account that posts only a few times leaves too few gaps to estimate a distribution from.

## `temp_median_interarrival_seconds`

Median time between consecutive posts.

- **Level**: account · **Unit**: seconds · **Direction**: lower = more anomalous
- **Rationale**: Sustained short gaps put an account outside what a person maintains by hand over a long collection window, especially combined with a flat circadian shape and an absent quiet window.
- **Known limitation**: Confounded with volume and with topic: live-posting an election night or a football match produces minutes-long medians from an entirely human account.

## `temp_burst_share`

Share of posts arriving inside a burst, relative to the account's baseline.

- **Level**: account · **Unit**: ratio · **Direction**: higher = more anomalous
- **Rationale**: Queued or triggered publishing tends to discharge in clumps far faster than the account's own median rhythm, rather than arriving at the irregular pace of someone typing.
- **Known limitation**: Thread writing, live commentary and catching up after a flight all look like bursts, and the threshold is relative, so a quiet account bursts more easily.

## `temp_max_burst_size`

Number of posts in the account's largest burst.

- **Level**: account · **Unit**: count · **Direction**: higher = more anomalous
- **Rationale**: Distinguishes an account that occasionally posts twice in a row from one that empties a queue of dozens of items in a few minutes.
- **Known limitation**: A long thread published in one sitting is a single large burst and is completely ordinary human behaviour on most platforms.

## `temp_active_days_ratio`

Share of days in the observed span on which the account posted.

- **Level**: account · **Unit**: ratio · **Direction**: higher = more anomalous
- **Rationale**: People take days off: they travel, get busy, lose interest for a week. A process that never misses a day over a long span is behaving unlike its audience even when its daily volume is modest.
- **Known limitation**: Professional accounts -- newsrooms, institutions, anyone whose job is to post -- reach 1.0 legitimately, and a short collection window reaches it by accident.

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
