"""Pseudonymisation for exported reports.

A report is the artefact that leaves the analyst's machine. It gets forwarded,
screenshotted and quoted, usually without the corpus that would let a reader
check anything, so it is the last place where a handle should appear next to a
number that looks like an accusation.

Replacing account ids with stable pseudonyms keeps a report readable -- the
same account is recognisably the same account across clusters and sections --
while making it useless as a target list on its own. Anyone who holds the
corpus can recompute the mapping in a second, which is the point: the people
who can already identify the accounts lose nothing, and everyone else gains
nothing from the file.

This is not anonymisation. A cluster card describing four accounts that posted
identical text at a known minute is re-identifiable by anyone with access to
the platform. It is a speed bump against casual misuse, and it should be
described as exactly that.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

from synthwatch.types import AccountId

__all__ = ["Pseudonymiser"]

PSEUDONYM_LENGTH = 10
"""Hex characters kept from the digest. Collisions are checked, not assumed."""


@dataclass
class Pseudonymiser:
    """Maps account ids to stable, salted pseudonyms.

    Args:
        salt: Secret mixed into every digest. Generated randomly when omitted,
            which makes two reports of the same corpus use different
            pseudonyms. Pass an explicit salt when reports need to be
            comparable -- and record that decision, because a shared salt makes
            the pseudonyms joinable across every report that used it.
        enabled: When ``False`` the mapping is the identity. Useful for an
            analyst working on their own machine; the exported artefact is
            where the default matters.

    Note:
        The mapping is deliberately never serialised by the report layer. It
        lives for the life of this object and nothing writes it to disk.
    """

    salt: str = field(default_factory=lambda: secrets.token_hex(16))
    enabled: bool = True
    _forward: dict[AccountId, str] = field(default_factory=dict, repr=False)
    _reverse: dict[str, AccountId] = field(default_factory=dict, repr=False)

    def __call__(self, account_id: AccountId) -> str:
        """Return the pseudonym for ``account_id``, creating it if needed."""
        if not self.enabled:
            return account_id
        cached = self._forward.get(account_id)
        if cached is not None:
            return cached

        pseudonym = self._digest(account_id, PSEUDONYM_LENGTH)
        # Two accounts sharing a pseudonym would silently merge in a report, so
        # widen the digest until it is unique rather than hoping it will be.
        width = PSEUDONYM_LENGTH
        while pseudonym in self._reverse:
            width += 2
            pseudonym = self._digest(account_id, width)

        self._forward[account_id] = pseudonym
        self._reverse[pseudonym] = account_id
        return pseudonym

    def _digest(self, account_id: AccountId, width: int) -> str:
        """Salted digest of an account id, truncated to ``width`` characters."""
        digest = hashlib.blake2b(
            account_id.encode("utf-8"), key=self.salt.encode("utf-8"), digest_size=16
        ).hexdigest()
        return f"acct_{digest[:width]}"

    def resolve(self, pseudonym: str) -> AccountId | None:
        """Look up the account behind a pseudonym, for the analyst who made it.

        Only works for pseudonyms this object has already issued, and only in
        memory. It exists so an analyst can follow up on a cluster in their own
        session, not so a report can be reversed by its readers.
        """
        return self._reverse.get(pseudonym)

    def __len__(self) -> int:
        """How many accounts have been given a pseudonym."""
        return len(self._forward)
