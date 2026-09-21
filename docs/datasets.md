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

**The defect you must report:** this dataset has **one class**. Every account in
it was removed. A model trained on it plus a control group collected some other
way learns to tell the two *collections* apart and reports a superb score for
doing so. `label_coverage` warns about this at load time, and the ensemble
refuses to train on it alone.

`INFO_OPERATION` is not a synonym for `AUTOMATED` — many of these accounts were
run by people, full time, by hand.

---

## Labelled: Indiana University Bot Repository

Annotation files — account id plus a class — from a series of studies, with the
posts distributed separately or not at all.

- <https://botometer.osome.iu.edu/bot-repository/datasets.html>
- Some datasets download directly; others link to the original authors and need
  a request. Expect to email a few researchers.

```bash
synthwatch labels corpus.json varol-2017.dat \
    --dataset indiana-bot-repository/varol-2017
```

**The defects you must report:**

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
