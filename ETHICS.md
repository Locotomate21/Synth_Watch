# Scope and limits

`synthwatch` is a measurement instrument for research on information
environments. It is not a moderation system, an attribution system, or a
surveillance tool, and several design decisions exist specifically to keep it
from turning into one.

## What the library refuses to produce

1. **No individual binary verdicts.** The API never returns "this account is a
   bot". Account-level outputs are calibrated probabilities with an explicit
   uncertainty interval, and the report layer publishes them only in
   aggregate: corpus-level prevalence and cluster-level cards.
2. **No verdict fields on records.** `Account` and `Post` have no `is_bot`,
   `label` or `score` attribute, so no serialised artefact of this library can
   circulate as an accusation. A test enforces it.
3. **No identity resolution.** Nothing here links accounts to real-world
   people, cross-references platforms by person, or geolocates anyone.
4. **No collection tooling.** Adapters read files. Fetching, scraping and
   credential handling are out of scope.

## Known error modes that must travel with any result

- **Base rates are unknown.** A classifier trained on the Indiana Bot
  Repository or the Twitter Information Operations Archive learns the
  automation of one platform in one era. Prevalence estimates transfer badly
  across platforms, languages and years; treat cross-domain numbers as ordinal
  at best.
- **Coordination is not intent.** Fans, activists, newsrooms, hobbyists and
  ordinary scheduling tools all produce near-simultaneous near-duplicate
  content. A dense cluster is a question to investigate, not a finding.
- **Synthetic-text detection is unreliable on short text, on non-English text,
  and on text written by non-native speakers.** Perplexity-based signals
  systematically flag second-language authors. Post-level synthetic scores are
  therefore never exported individually.
- **Labels are noisy.** Suspension-derived labels encode a platform's
  enforcement policy, not ground truth. `LabelRecord.method` records which kind
  of label an evaluation relied on, and it should be reported.

## Publication guidance

Report the corpus, the collection window, the adapter, the drop rate, the model
version and the calibration set alongside any number this library produces. A
prevalence figure without them is not a result.
