# Where to get real data

Two different needs, met by different sources.

**Unlabelled political conversation** is what `analyse` works on: coordination
and posting rhythm are unsupervised, so a corpus with no labels at all already
produces a report.

**Labelled accounts** are what `train` needs, and they are the scarce resource.
Every public option has a defect that has to be reported alongside any result.

---

## Labelled: X / Twitter Information Operations Archive

Every tweet and profile from accounts a platform removed and attributed to a
state-linked operation.

- **Mirror (recommended):** <https://archive.org/details/X_Twitter_Information_Operations>
  — `ioa_users.csv` (15.7 MB) and `ioa_tweets.csv` (105.9 GB), also available
  as a torrent. Mirrored in January 2024 because X's own transparency portal
  became unreliable.
- **Original:** X's Moderation Research Consortium, which historically required
  an email address and has been intermittent since.

Verified against the real file:

```bash
synthwatch inspect ioa_users.csv --kind accounts \
    --alias account_creation_date=created_at --date-order mdy --assume-timezone 0
```

Measured on the full file — 87,377 account rows, of which **84,972 load and
2,405 do not**, every one of them explained:

```
trial load of the sample: 84972 record(s)
  skipped 2315: account:corrupted_identifier
  skipped 89: account:empty_row
  skipped 1: account:repeated_header
```

Four things it turns up, which the loaders now handle explicitly:

| What | Why it matters |
| --- | --- |
| `account_creation_date` reads `7/3/2018` | Either 7 March or 3 July. Refused unless `--date-order` is declared. |
| Some `userid` values read `1.01421E+18` | A spreadsheet round-trip destroyed an 18-digit id. **2,315 rows of the real file — 2.6%.** They join to nothing and are counted as skipped. |
| Timestamps carry no offset | The archive documents them as UTC; `IOArchiveAdapter` declares that once rather than guessing per row. |
| One row repeats the header | The consolidated file was built by concatenating per-takedown exports without stripping their headers. |
| 97% of handles are a hash | The archive anonymises every account below its follower threshold by rewriting `userid`, `user_display_name` and `user_screen_name` to one digest. |

### Getting the tweets without downloading 113 GB

`ioa_tweets.csv` is 113.72 GB, and `archive.org` answers `Accept-Ranges: bytes`,
so a slice can be pulled directly over HTTP.

**A prefix is not a random sample.** The file is a concatenation of per-takedown
exports, so it is grouped by campaign. Probing it at intervals:

| offset | what is there |
| ---: | --- |
| 0 GB | Bangladesh |
| 10 GB | English |
| 56 GB | Arabic |
| 70–85 GB | Serbian (Belgrade) |
| 105 GB | Turkish (İstanbul) |
| 108 GB | Uganda |
| **110 GB – end** | **Spanish (Venezuela)** |

Downloading "the first 2 GB" gets Bangladesh. For Spanish-language political
conversation, the last ~3.7 GB is the part that matters.

A ranged download starts and ends mid-line and carries no header, so all three
have to be handled:

```bash
URL=https://archive.org/download/X_Twitter_Information_Operations/ioa_tweets.csv

# the header, from the first line of the file
curl -sL -r 0-600 "$URL" | head -1 > header.csv

# a 500 MB slice from the Spanish-language region
curl -L -r 110000000000-110500000000 "$URL" -o slice.raw

# drop the truncated first and last lines, prepend the header
{ cat header.csv; tail -n +2 slice.raw | head -n -1; } > data/raw/io_tweets_es.csv
```

Measured: roughly **2,000 posts per MB**, so 500 MB is about a million posts and
the full Spanish region about 7.7 million. A 20 MB test slice loaded 41,668
posts with nothing skipped.

Only one declaration is needed — `--assume-timezone 0`, since the archive's
timestamps carry no offset.

There is also a torrent on the item page, which is the better option for the
whole file.

**The defect you must report:** this dataset has **one class**. Every account in
it was removed. A model trained on it plus a control group collected some other
way learns to tell the two *collections* apart and reports a superb score for
doing so. `label_coverage` warns about this at load time, and the ensemble
refuses to train on it alone.

`INFO_OPERATION` is not a synonym for `AUTOMATED` — many of these accounts were
run by people, full time, by hand.

### The anonymisation is the bigger trap

Accounts below the archive's follower threshold have their id, display name and
screen name replaced by a single hash. That is 97% of the file, and it means
the handle a naive loader reads is a base64 digest.

Computed over that, the handle features describe the *anonymisation*: a digest
has the digit ratio and character entropy of a digest. The first run of this
pipeline against the file reported a handle-entropy median of 0.96 at 100%
coverage, which is the entropy of base64 and says nothing at all about how the
accounts were named. Profile completeness was inflated the same way, because a
display name set to the account's own hash counted as a filled-in field.

`has_pseudonymised_identity` now detects the pattern — any identity field equal
to the account id — and the affected features are withheld. The honest picture:

| feature | coverage | p10 | median | p90 |
| --- | ---: | ---: | ---: | ---: |
| `acct_age_days` | 100% | 1450 | 1820 | 3530 |
| `acct_followback_ratio` | 75% | 0.00 | 0.33 | 0.80 |
| `acct_profile_completeness` | 100% | 0.00 | 0.00 | 0.67 |
| `acct_handle_digit_ratio` | **3%** | 0.00 | 0.00 | 0.33 |
| `acct_handle_entropy` | **3%** | 0.94 | 0.97 | 1.00 |

The coverage column is the point: 3% is the share of the dataset whose handles
can be read at all. Profile completeness also moved — its median fell from 0.25
to 0.00 once the hashed display name stopped counting, which means half of
these accounts filled in nothing whatsoever.

The file also carries **240 duplicate account ids** across its 84,972 rows.

---

## Labelled: accounts with a negative class

The X archive has only removed accounts. Anything that trains a model needs the
other side, and that is where most of the historical links have rotted.

### Verified working (September 2026)

**`airt-ml/twitter-human-bots`** on Hugging Face — 37,438 accounts, 12,425 bot
and 25,013 human, one CSV, no access request.

```bash
curl -L -o data/raw/twitter_human_bots.csv   https://huggingface.co/datasets/airt-ml/twitter-human-bots/resolve/main/twitter_human_bots_dataset.csv
```

Its columns map almost entirely onto the schema: `id`, `screen_name`,
`created_at`, `description`, `followers_count`, `friends_count`,
`statuses_count`, `verified`, `location`, plus `account_type` as the class.
Only one declaration is needed — `created_at` carries no offset, and Twitter
published those in UTC.

### Requires an application

- **TwiBot-22** — <https://github.com/LuoUndergradXJTU/TwiBot-22>. One million
  accounts with a graph structure. Access by emailing the authors from an
  institutional address, stating your institution, advisor and use case.
- **TwiBot-20** — <https://github.com/BunsenFeng/TwiBot-20>. Smaller, same
  arrangement.

### Gone

**The Indiana University Bot Repository is no longer published.**
`botometer.osome.iu.edu/bot-repository/datasets.html` now returns a soft 404 —
the server answers 200 with the site shell and the client renders "page not
found" — and Botometer itself has been rebuilt to score Bluesky accounts. Any
guide still pointing there, including earlier versions of this file, is stale.

```bash
synthwatch labels accounts.json labels.tsv \
    --dataset airt-ml/twitter-human-bots
```

**The defects you must report,** whichever source you use:

- **The class vocabulary differs per file** — `bot`/`human` in one,
  `social_spambot_1` in another, `0`/`1` in a third. Numeric files are refused
  unless you declare which way round they run, because both conventions are
  published and picking one inverts the training set.
- **Annotations are older than the accounts.** Many ids no longer resolve, and
  the survivors are not a random sample. `label_coverage` reports the match rate
  and warns below 50%.
- **Procedures differ between datasets.** Pooling them mixes their error modes;
  the coverage report says so when it sees more than one `--dataset`.

---

## Unlabelled: conversation to analyse

- **Bluesky** — open firehose, and there are ready-made dumps on Hugging Face
  ([two million posts](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts),
  [five million](https://huggingface.co/datasets/Roronotalt/bluesky-five-million)).
  No API key, no gatekeeping, current. Best starting point for `analyse`.
- **Mastodon** — public timelines are readable per instance; good for a small,
  well-scoped corpus you collect yourself.
- **Reddit** — Pushshift-style historical dumps via Academic Torrents. Large,
  and the political subreddits are the obvious target.

These carry no labels, which is fine: coordination, posting rhythm and the
profile features are all unsupervised. What they cannot do is train the
ensemble.

---

## What the pipeline needs from a file

Minimum for a post: **an id, an author id, and a timestamp**. Everything else
improves the analysis and nothing else is required.

| Schema field | Needed for |
| --- | --- |
| `post_id`, `account_id`, `created_at` | everything |
| `text` | coordination (near-duplicate detection) |
| `root_post_id` | keeping one busy thread from reading as coordination |
| `kind` / a retweet flag | excluding reshares from the co-posting graph |
| `created_at` on the **account** | age and dormancy |
| `followers_count`, `following_count` | the network ratio |
| `handle` | the handle-shape features |

Point `inspect` at the file first. It samples a couple of thousand rows and
reports which columns map, which land in `extra`, which required field is
missing, and exactly what a load would skip and why — without reading a
hundred gigabytes to find out.

```bash
synthwatch inspect whatever_you_downloaded.csv
```

---

## The first model this pipeline trained

Against `airt-ml/twitter-human-bots`, 37,438 accounts, all labels matched:

```
roc_auc 0.872  average_precision 0.800
brier 0.130  calibration error 0.005
negative: precision 0.83 recall 0.91
positive: precision 0.77 recall 0.64
```

The calibration is the part worth looking at, because it is what the
probabilities claim to mean:

| predicted | observed | n |
| ---: | ---: | ---: |
| 0.04 | 0.04 | 9,754 |
| 0.25 | 0.25 | 4,770 |
| 0.55 | 0.55 | 1,656 |
| 0.75 | 0.77 | 2,518 |
| 0.96 | 0.97 | 2,560 |

Of the accounts this model called 25% likely, 25% were labelled bots. That
holds across all ten bins.

What the model is actually reading, by mean absolute SHAP value:

| feature | importance |
| --- | ---: |
| `acct_followback_ratio` | 0.809 |
| `acct_posts_per_day` | 0.541 |
| `acct_age_days` | 0.344 |
| `acct_profile_completeness` | 0.261 |
| `acct_handle_digit_ratio` | 0.109 |
| `acct_handle_trailing_digits` | 0.048 |
| `acct_handle_entropy` | 0.028 |

**Read this as a floor, not a result.** The dataset carries no posts, so 15 of
the 22 features were unmeasurable and dropped — every temporal and coordination
signal among them. The model card says so. This is what the profile features
alone can do, and the question the pipeline was built to answer needs the rest.

---

## The first coordination analysis this pipeline ran

306,434 posts from the Spanish-language region of the archive, spanning
2010-02-11 to 2015-12-13, at the default 15-minute window.

```
graph        : 298 accounts, 2,668 edges
eligible     : 289,339 posts   matched: 34,677
null model   : 2,668 observed edges vs 0.15 by chance   p = 0.0476
```

The null model is the number that matters. Time-shifting each account's
timeline independently — keeping every account's own rhythm and destroying only
the alignment *between* them — leaves an average of **0.15 edges**. The observed
graph has 2,668. The p-value sits at 1/21 because that is the floor with 20
permutations, not because the evidence is marginal.

| cluster | accounts | density | mean similarity | median lag | co-posts |
| --- | ---: | ---: | ---: | ---: | ---: |
| c000 | 50 | 0.92 | 0.998 | 0 s | 25,910 |
| c001 | 42 | 0.07 | 0.992 | 180 s | 706 |
| c002 | 32 | 0.49 | 0.974 | 0 s | 15,703 |
| c003 | 10 | 1.00 | 0.997 | 0 s | 753 |
| c004 | 3 | 1.00 | 1.000 | 0 s | 709 |

### What the window sweep said, and why it was the opposite of the worry

`detect.coordination` warns that the window is the finding — that widening it
from 5 to 60 minutes can turn a null result into a dense graph. Swept across a
240-fold range:

| window | edges | clusters | accounts in clusters |
| ---: | ---: | ---: | ---: |
| 1 min | 2,515 | 8 | 117 |
| 5 min | 2,594 | 8 | 129 |
| 15 min | 2,668 | 5 | 137 |
| 60 min | 2,779 | 6 | 149 |
| 240 min | 2,900 | 5 | 156 |

Going from one minute to four hours adds 15% more edges. Almost every pair is
**already inside the first minute**, which is what a median lag of 0 seconds
means at this file's minute-resolution timestamps. The result does not depend
on the parameter, and the sweep is what establishes that rather than asserting
it.

None of this says who or why. It says fifty accounts published near-identical
text in the same minute, tens of thousands of times, and that chance does not
produce it.
