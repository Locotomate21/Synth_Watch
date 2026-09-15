"""Synthetic scenarios for the coordination module.

Nothing here is collected data. Each builder plants a known structure so a test
can assert both what the method should find and what it must refuse to find.
The Spanish text is deliberate: the module must not depend on English-specific
tokenisation. The text is ASCII-only so that a failing assertion prints cleanly
on a Windows console.
"""

from __future__ import annotations

from synthwatch.models import Corpus, Post
from synthwatch.types import PostKind
from tests.conftest import make_corpus, make_post

TEMPLATES = (
    "La reforma que aprobaron de madrugada no arregla el problema de fondo, deja fuera "
    "otra vez a la gente que mas lo necesitaba y encima la pagamos entre todos. No es "
    "un error de calculo, es una decision politica que alguien tomo sabiendo el efecto.",
    "Mientras discuten en el congreso sobre el presupuesto, en mi barrio llevamos tres "
    "semanas sin transporte decente, sin respuesta del ayuntamiento y sin nadie que se "
    "haga responsable. Que alguien explique donde ha ido a parar el dinero prometido.",
    "Cada vez que anuncian inversion en sanidad publica termina pasando exactamente lo "
    "mismo: mucho titular, mucha foto de la visita al hospital, y despues los mismos "
    "recortes de siempre en las plantillas que sostienen el servicio todo el ano.",
)
"""Long enough to clear ``min_chars``, far enough apart not to match each other."""

TWEAKS = ("", " !!", " ...", " ?", " !")
"""The cosmetic edits template posting uses to dodge exact-match dedup."""

ORGANIC_TEXTS = (
    "Hoy he conseguido por fin arreglar la bicicleta despues de dos meses parada en el trastero.",
    "Alguien sabe si la biblioteca del centro abre los domingos por la manana o solo por la tarde?",
    "Me he pasado la tarde intentando que el gato entre en el transportin y sigo sin conseguirlo.",
    "El pan de la panaderia nueva esta bastante bien, pero cierran demasiado pronto entre semana.",
    "Acabo de terminar una novela larguisima y ahora mismo no se que leer, acepto recomendaciones.",
    "Llueve otra vez y yo me deje el paraguas encima de la mesa de la oficina, cosas que pasan.",
    "Han puesto un carril bici nuevo junto al rio y la verdad es que se agradece un monton.",
    "No entiendo como puede costar tanto encontrar un fontanero un sabado por la tarde en agosto.",
    "Llevo toda la semana intentando cambiar una cita medica por telefono y nadie me lo coge.",
    "El concierto de ayer empezo con hora y media de retraso pero valio bastante la pena.",
    "Me han cobrado dos veces el mismo recibo y el banco dice que tardan diez dias en revisarlo.",
    "Estoy aprendiendo a hacer pan en casa y de momento el resultado parece mas un ladrillo.",
)
"""One distinct text per organic post, so nothing matches by accident."""


def coordinated_posts(
    prefix: str = "sync",
    *,
    n_accounts: int = 4,
    n_waves: int = 3,
    wave_gap_minutes: float = 240.0,
    jitter_seconds: float = 40.0,
    start_minutes: float = 0.0,
) -> list[Post]:
    """Accounts pushing the same templates in tight, repeated waves.

    Each wave puts one template in every timeline within
    ``n_accounts * jitter_seconds``, with a cosmetic tweak per account. Waves
    are hours apart, so the repetition across waves is what builds edge weight
    rather than one burst being counted several times.
    """
    posts: list[Post] = []
    for wave in range(n_waves):
        template = TEMPLATES[wave % len(TEMPLATES)]
        wave_start = start_minutes + wave * wave_gap_minutes
        for account in range(n_accounts):
            posts.append(
                make_post(
                    f"{prefix}{account}_w{wave}",
                    f"{prefix}{account}",
                    offset_minutes=wave_start + account * jitter_seconds / 60.0,
                    text=template + TWEAKS[account % len(TWEAKS)],
                )
            )
    return posts


def organic_posts(prefix: str = "org", *, n_accounts: int = 6) -> list[Post]:
    """Unrelated accounts writing unrelated things at unrelated times."""
    return [
        make_post(
            f"{prefix}{account}_p{index}",
            f"{prefix}{account}",
            offset_minutes=account * 97.0 + index * 613.0,
            text=ORGANIC_TEXTS[account * 2 + index],
        )
        for account in range(n_accounts)
        for index in range(2)
    ]


def thread_posts(prefix: str = "thr", *, n_accounts: int = 5) -> list[Post]:
    """One busy thread: everybody quoting the same message, minutes apart.

    The honest shape of an argument, and the commonest false positive for any
    co-posting method that ignores conversation structure.
    """
    quoted = TEMPLATES[0]
    return [
        make_post(
            f"{prefix}{account}_r{turn}",
            f"{prefix}{account}",
            offset_minutes=turn * 6.0 + account * 1.1,
            text=quoted + TWEAKS[account % len(TWEAKS)],
            kind=PostKind.REPLY,
            parent_post_id=f"{prefix}_root",
            root_post_id=f"{prefix}_root",
        )
        for turn in range(2)
        for account in range(n_accounts)
    ]


def headline_posts(prefix: str = "news", *, n_accounts: int = 4) -> list[Post]:
    """Unconnected accounts pasting the same breaking-news headline.

    The documented false positive. The method is expected to flag this, which
    is precisely why cluster cards carry a caveat instead of a verdict.
    """
    headline = (
        "ULTIMA HORA: el tribunal anula la sentencia y ordena repetir el juicio por "
        "defectos de forma en la instruccion del caso, segun fuentes juridicas citadas "
        "esta manana por varios medios nacionales."
    )
    return [
        make_post(
            f"{prefix}{account}_h{wave}",
            f"{prefix}{account}",
            offset_minutes=wave * 300.0 + account * 0.8,
            text=headline,
        )
        for wave in range(2)
        for account in range(n_accounts)
    ]


def coordinated_corpus() -> Corpus:
    """A planted cluster of four accounts inside ordinary conversation."""
    return make_corpus([*coordinated_posts(), *organic_posts()], source="synthetic:coordinated")


def organic_corpus() -> Corpus:
    """Conversation with nothing planted in it."""
    return make_corpus(organic_posts(), source="synthetic:organic")


def thread_corpus() -> Corpus:
    """A single dense thread, plus unrelated chatter."""
    return make_corpus([*thread_posts(), *organic_posts()], source="synthetic:thread")


def headline_corpus() -> Corpus:
    """Accounts sharing the same headline at the same time."""
    return make_corpus(headline_posts(), source="synthetic:headline")
