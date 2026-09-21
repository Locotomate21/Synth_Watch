"""Profile-shaped signals: age, handle, network ratios, completeness, volume.

These are the cheapest features in the library and the ones to be most careful
with. They describe *how an account is set up*, not what it does, so every one
of them also describes a large population of ordinary people: newcomers,
people who never filled in a bio, people who read far more than they post.
None of them is evidence of anything on its own, and the ensemble is where
they are allowed to matter -- in combination, and calibrated.

Two rules the module holds to:

**Missing metadata is missing, not zero.** A corpus without follower counts, or
without profile creation dates, is extremely common: some platforms do not
expose them, some exports drop them, and suspended accounts lose them. Every
feature here returns ``NaN`` when its inputs are absent. Reading an absent bio
as an empty bio would manufacture a signal out of a gap in the data.

**Nothing is measured against "now".** Age and dormancy are measured against
the corpus collection time, so re-running an analysis a year later does not
silently age every account in it.

Known limitations
-----------------

* **Handle shape is culturally loaded.** Digits in a handle mean a birth year
  in one place, a jersey number in another, and a platform's collision
  suffix in a third. The digit features carry real signal in aggregate and
  are close to worthless for any individual account.
* **Archives anonymise, and anonymisation looks like data.** Where a source
  has replaced an account's handle and display name with a hash of its id --
  97% of the rows in the published Information Operations archive -- the
  handle features would measure the digest rather than the account.
  :func:`has_pseudonymised_identity` detects that and the features are
  withheld, so their coverage figure reports how much of the corpus could
  actually be read.
* **Young accounts are mostly just young.** Every platform has a constant
  inflow of genuine new users, and any campaign with a budget buys aged
  accounts precisely to defeat this feature.
* **Network ratios reflect what a platform rewards.** Follow-back norms differ
  enormously between Mastodon, Reddit and Bluesky, so the same ratio means
  different things across the corpora this library is meant to compare.
* **Platform-reported volume is not corpus volume.** ``post_count`` is a
  lifetime total from the profile snapshot, while the corpus holds whatever
  was collected. They are different quantities and the features keep them
  apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from synthwatch.detect.base import FeatureSpec, empty_frame
from synthwatch.detect.stats import character_entropy, safe_ratio
from synthwatch.models import Account, Corpus
from synthwatch.types import AccountId, FeatureLevel

__all__ = [
    "AccountConfig",
    "AccountExtractor",
    "available_profile_fields",
    "digit_ratio",
    "handle_shape",
    "has_pseudonymised_identity",
    "profile_completeness",
]

PROFILE_FIELDS: tuple[str, ...] = (
    "display_name",
    "description",
    "location",
    "url",
)
"""Optional profile fields counted by :func:`profile_completeness`.

Deliberately excludes anything a platform fills in by itself, so the measure
reflects effort spent by whoever set the account up.
"""


@dataclass(frozen=True, slots=True)
class HandleShape:
    """What a handle looks like, independent of what it says."""

    length: int
    digit_ratio: float
    trailing_digits: int
    entropy: float


def digit_ratio(handle: str) -> float:
    """Share of characters in ``handle`` that are digits."""
    if not handle:
        return 0.0
    return sum(character.isdigit() for character in handle) / len(handle)


def trailing_digits(handle: str) -> int:
    """Length of the run of digits at the end of ``handle``.

    A long trailing run is the signature of an automatically suggested or
    generated name -- the platform appending enough digits to make a taken
    name unique, or a script doing the same thing deliberately.
    """
    count = 0
    for character in reversed(handle):
        if not character.isdigit():
            break
        count += 1
    return count


def handle_shape(handle: str) -> HandleShape:
    """Measure the shape of a handle."""
    return HandleShape(
        length=len(handle),
        digit_ratio=digit_ratio(handle),
        trailing_digits=trailing_digits(handle),
        entropy=character_entropy(handle.casefold()),
    )


def has_pseudonymised_identity(account: Account) -> bool:
    """Whether the source replaced this account's identity with its id.

    Archives anonymise. The published Information Operations archive rewrites
    ``userid``, ``user_display_name`` and ``user_screen_name`` to one hash for
    every account below a follower threshold -- 97% of its rows -- so the
    handle it ships is a base64 digest, not a name.

    Computing handle shape over that measures the anonymisation: a digest has
    the digit ratio and the character entropy of a digest, and the resulting
    distribution says nothing whatever about how the accounts were named. The
    features are therefore withheld rather than computed, which is the same
    distinction the rest of the library keeps between "measured" and "could
    not measure".

    The test is deliberately narrow: a field that *equals the account id* is
    the identifier repeated, not a value. That is true whether the id is a
    hash, a number or anything else.
    """
    return account.account_id in {account.handle, account.display_name}


def profile_completeness(
    account: Account,
    fields: tuple[str, ...] = PROFILE_FIELDS,
    *,
    count_avatar: bool = True,
) -> float | None:
    """Share of the counted profile fields that carry something.

    Args:
        account: The account to measure.
        fields: Exactly the fields to count. The caller decides which ones the
            source actually exposes; see :func:`available_profile_fields`.
        count_avatar: Whether a recorded default avatar counts against the
            score.

    Returns:
        The share, or ``None`` when there is nothing to count at all -- which
        means the source exposes no profile metadata, not that the profile is
        empty.
    """
    counted = [bool(getattr(account, name, None)) for name in fields]
    if count_avatar and account.has_default_profile_image is not None:
        counted.append(not account.has_default_profile_image)
    if not counted:
        return None
    return sum(counted) / len(counted)


def available_profile_fields(
    corpus: Corpus, fields: tuple[str, ...] = PROFILE_FIELDS
) -> tuple[str, ...]:
    """Which profile fields this source exposes at all.

    Availability is a property of the *corpus*, not of an account. A single
    account with an empty bio has left it empty; a corpus where no account has
    a bio is a source that does not carry bios, and counting that absence
    against every account would manufacture a corpus-wide signal out of a gap
    in the export.
    """
    return tuple(
        name
        for name in fields
        if any(getattr(account, name, None) is not None for account in corpus.accounts)
    )


@dataclass(frozen=True, slots=True)
class AccountConfig:
    """Parameters of the profile analysis.

    Attributes:
        reference: Instant that age and dormancy are measured against. Leave
            it ``None`` to use the corpus collection time, falling back to the
            last post in the corpus. Never "now": an analysis re-run next year
            must produce the same numbers.
        profile_fields: Optional profile fields counted as completeness.
        min_handle_length: Handles shorter than this get ``NaN`` for the shape
            features, since a three-character handle has no measurable shape.
    """

    reference: datetime | None = None
    profile_fields: tuple[str, ...] = PROFILE_FIELDS
    min_handle_length: int = 4

    def reference_for(self, corpus: Corpus) -> datetime | None:
        """Resolve the instant to measure ages against for ``corpus``."""
        if self.reference is not None:
            return self.reference
        if corpus.collected_at is not None:
            return corpus.collected_at
        collected = [p.collected_at for p in corpus.posts if p.collected_at is not None]
        if collected:
            return max(collected)
        span = corpus.time_span
        return span[1] if span else None

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, meant to be embedded in every report."""
        return {
            "reference": self.reference.isoformat() if self.reference else None,
            "profile_fields": list(self.profile_fields),
            "min_handle_length": self.min_handle_length,
        }


SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="acct_age_days",
        description="Account age in days at the corpus collection time.",
        rationale=(
            "Accounts created for a specific campaign cluster in age around the event "
            "they were created for, and a cohort of accounts that all appeared in the "
            "same fortnight is a stronger signal than any single young account."
        ),
        limitation=(
            "Every platform has a constant inflow of genuine newcomers, and anyone with "
            "a budget buys aged accounts precisely to defeat this. Young means young."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="days",
        value_range=(0.0, None),
        higher_is_more_anomalous=False,
    ),
    FeatureSpec(
        name="acct_handle_digit_ratio",
        description="Share of the handle made up of digits.",
        rationale=(
            "Bulk-registered accounts take whatever name the platform suggests, which is "
            "a desired name plus enough digits to make it unique. Registering by hand, "
            "people usually keep trying until they find something they like."
        ),
        limitation=(
            "Birth years, jersey numbers and area codes are ordinary in handles, and in "
            "several languages a numeric suffix is a normal naming convention."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="acct_handle_trailing_digits",
        description="Length of the run of digits at the end of the handle.",
        rationale=(
            "A long trailing digit run is specifically the shape of an auto-suggested "
            "name, which distinguishes it from a birth year or a meaningful number "
            "embedded inside a handle."
        ),
        limitation=(
            "Four trailing digits are just as likely to be a year of birth, which makes "
            "the low end of this feature almost uninformative on its own."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="count",
        value_range=(0.0, None),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="acct_handle_entropy",
        description="Normalised character entropy of the handle.",
        rationale=(
            "Handles drawn from a random generator spread evenly across their character "
            "set, while names chosen by people reuse letters and follow the statistics "
            "of a language."
        ),
        limitation=(
            "Short handles score high mechanically, transliterated names look random to "
            "a measure built around the Latin alphabet, and the feature says nothing at "
            "all about accounts whose handle the source did not record."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="bits_normalised",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="acct_followback_ratio",
        description="Followers as a share of followers plus following.",
        rationale=(
            "Accounts built to amplify follow aggressively and are followed back rarely, "
            "so they sit low on this scale. Expressing it as a share rather than a raw "
            "ratio keeps accounts with zero followers from producing infinities."
        ),
        limitation=(
            "Follow-back norms differ enormously between platforms, so the same value "
            "means different things across corpora; new genuine accounts also start low."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=False,
    ),
    FeatureSpec(
        name="acct_profile_completeness",
        description="Share of optional profile fields that were filled in.",
        rationale=(
            "Setting up a profile costs attention that scales badly across hundreds of "
            "accounts, so bulk-created accounts tend to leave the optional fields empty."
        ),
        limitation=(
            "Plenty of careful, private people never write a bio, and a well-funded "
            "operation fills every field precisely because this is a known check."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="ratio",
        value_range=(0.0, 1.0),
        higher_is_more_anomalous=False,
    ),
    FeatureSpec(
        name="acct_posts_per_day",
        description="Lifetime posts reported by the profile, divided by account age.",
        rationale=(
            "Sustained volume over the whole life of an account is hard to keep up by "
            "hand, and unlike the corpus-derived rate it is not limited to whatever "
            "slice of activity happened to be collected."
        ),
        limitation=(
            "It is a lifetime average, so a dormant account that was briefly frantic "
            "looks moderate, and a deleted backlog makes an active account look quiet."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="posts_per_day",
        value_range=(0.0, None),
        higher_is_more_anomalous=True,
    ),
    FeatureSpec(
        name="acct_dormancy_days",
        description="Days between account creation and its first post in the corpus.",
        rationale=(
            "Accounts registered long before they are used, then activated together, are "
            "the signature of an aged inventory. The gap is visible even when the age "
            "itself looks unremarkable."
        ),
        limitation=(
            "Only the *observed* first post is available, so an account that posted for "
            "years before the collection window looks dormant when it simply was not "
            "collected. It is meaningful only for corpora that reach back far enough."
        ),
        level=FeatureLevel.ACCOUNT,
        unit="days",
        value_range=(0.0, None),
        higher_is_more_anomalous=True,
    ),
)


class AccountExtractor:
    """Profile-shaped features, one row per account.

    Every feature is ``NaN`` when the metadata it needs is absent, which for
    this family is the common case rather than the exception. An account with
    no recorded handle is not an account with a plain handle.
    """

    name = "account"
    specs = SPECS

    def __init__(self, config: AccountConfig | None = None) -> None:
        self.config = config or AccountConfig()

    def extract(self, corpus: Corpus) -> pd.DataFrame:
        """Compute the profile features for every account in ``corpus``."""
        frame = empty_frame(corpus, self.specs)
        reference = self.config.reference_for(corpus)
        fields = available_profile_fields(corpus, self.config.profile_fields)
        count_avatar = any(a.has_default_profile_image is not None for a in corpus.accounts)
        first_posts: dict[AccountId, datetime] = {
            account_id: posts[0].created_at for account_id, posts in corpus.posts_by_account.items()
        }

        for account in corpus.accounts:
            if account.account_id not in frame.index:
                continue
            row = account.account_id
            age = account.age_days(reference) if reference else None
            if age is not None:
                frame.loc[row, "acct_age_days"] = age
                rate = safe_ratio(account.post_count, age)
                if rate is not None:
                    frame.loc[row, "acct_posts_per_day"] = rate

            measurable_handle = (
                account.handle
                and len(account.handle) >= self.config.min_handle_length
                and not has_pseudonymised_identity(account)
            )
            if measurable_handle and account.handle:
                shape = handle_shape(account.handle)
                frame.loc[row, "acct_handle_digit_ratio"] = shape.digit_ratio
                frame.loc[row, "acct_handle_trailing_digits"] = float(shape.trailing_digits)
                frame.loc[row, "acct_handle_entropy"] = shape.entropy

            reach = safe_ratio(
                account.followers_count,
                None
                if account.followers_count is None or account.following_count is None
                else account.followers_count + account.following_count,
            )
            if reach is not None:
                frame.loc[row, "acct_followback_ratio"] = reach

            # A display name that is really the account's own hash is not a
            # filled-in field, so it must not count towards completeness.
            countable = (
                tuple(name for name in fields if name != "display_name")
                if has_pseudonymised_identity(account)
                else fields
            )
            completeness = profile_completeness(account, countable, count_avatar=count_avatar)
            if completeness is not None:
                frame.loc[row, "acct_profile_completeness"] = completeness

            first = first_posts.get(account.account_id)
            if first is not None and account.created_at is not None:
                dormancy = (first - account.created_at).total_seconds() / 86_400.0
                frame.loc[row, "acct_dormancy_days"] = max(dormancy, 0.0)
        return frame
