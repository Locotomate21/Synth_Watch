# data/

Datasets live here and **nothing in this directory is committed** except this
file. Corpora describe real accounts, and a research repository is not a place
to redistribute them — the sources in [`../docs/datasets.md`](../docs/datasets.md)
are where they come from and where they should stay.

```
data/
├── raw/        exactly as downloaded, never edited
└── derived/    produced by the pipeline, reproducible from raw/
```

The split matters for one reason: anything in `derived/` can be deleted and
rebuilt, and anything in `raw/` cannot without downloading it again. Keeping
them apart is what makes it safe to clear the cache.

## What is here now

| File | Source |
| --- | --- |
| `raw/ioa_users.csv` | Profiles from the [Internet Archive mirror](https://archive.org/details/X_Twitter_Information_Operations) of X's information operations disclosures. 87,377 rows. |
| `derived/io_accounts.json` | The above, normalised into the internal schema. 84,972 accounts; the 2,405 rejected rows and their reasons are in `docs/datasets.md`. |
| `derived/io_account_features.csv` | The account feature matrix. **Per-account values — this is not a report** and does not leave the machine. |
| `raw/twitter_human_bots.csv` | 37,438 labelled accounts (12,425 bot, 25,013 human) from [airt-ml/twitter-human-bots](https://huggingface.co/datasets/airt-ml/twitter-human-bots). The negative class the X archive lacks. |
| `derived/human_bots.json` | The above, normalised. No rows rejected. |
| `derived/human_bots_labels.tsv` | Its `account_type` column, as an annotation file. |
| `derived/model_card.json` | The trained model's card: metrics, calibration curve, dropped features, warnings. |
| `raw/io_tweets_es.csv` | A 500 MB range slice of `ioa_tweets.csv` from 110 GB in — the Spanish-language region. 1,098,942 posts, nothing skipped. See `docs/datasets.md` for the campaign map and the slicing recipe. |
| `raw/io_tweets_es_400k.csv` | The first 400,000 lines of the above. A million posts needs about 7.5 GB of RAM; this fits. |

## Rebuilding derived/

```bash
synthwatch inspect data/raw/ioa_users.csv --kind accounts \
    --alias account_creation_date=created_at --date-order mdy --assume-timezone 0
```

Both declarations are required and neither is a default: the archive writes
`7/3/2018`, which is either March or July, and its timestamps carry no offset.
The library refuses both rather than guessing.

## Memory

Posts are pydantic records, and they are not small. Measured: **270,000 posts
occupy about 1.9 GB**, so a million needs roughly 7.5 GB and will be killed on
a 16 GB machine that is doing anything else. A first run on a few hundred
thousand posts says everything a larger one would about whether the pipeline
works; scale after, deliberately.

## Still needed

Labels that reach the *posts*. `twitter_human_bots.csv` labels accounts but
carries no posts; `io_tweets_es.csv` carries posts but every account in it was
removed, so it is single-class. Nothing here yet lets a model train on the
temporal and coordination features, which is the half of the pipeline the
account features cannot stand in for.

The Indiana Bot Repository, which earlier versions of these docs pointed at, is
no longer published.
