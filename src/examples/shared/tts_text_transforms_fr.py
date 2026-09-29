# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""French counterparts of Pipecat's English-biased TTS text transforms.

Pipecat's ``pipecat.utils.text.transforms`` hard-code English words, ``num2words(lang="en")``
and US conventions (``.`` decimal, ``,`` thousands, ``MM/DD/YYYY``, NANP phone numbers).
The transforms below mirror their names and signature
(``async def transform(text: str, aggregation_type: str) -> str``) for French text:
``,`` decimal separator, space / NBSP / narrow-NBSP thousands separators, ``DD/MM/YYYY``
dates, French phone numbers, and ``15h30`` style times (which Pipecat does not cover).
Plain integers and ordinals are left to the TTS server's text normalization, which handles them.
"""

import re

from num2words import num2words
from pipecat.frames.frames import AggregationType

# Integer part with optional space / NBSP / narrow-NBSP thousands groups, optional ``,`` decimals.
_NUMBER = r"(?<![\w.,])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:,(\d+))?"


def _to_number(whole_str: str, frac_str: str | None) -> int | float:
    whole = int(re.sub(r"\D", "", whole_str))
    return float(f"{whole}.{frac_str}") if frac_str else whole


def _number_words(whole_str: str, frac_str: str | None) -> str:
    return num2words(_to_number(whole_str, frac_str), lang="fr")


# --- Currency ----------------------------------------------------------------

_CURRENCY_MAP: dict[str, tuple[str, str, str | None, str | None]] = {
    "€": ("euro", "euros", "centime", "centimes"),
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("livre", "livres", "penny", "pence"),
    "¥": ("yen", "yens", None, None),
    "₹": ("roupie", "roupies", "paisa", "paisa"),
}

_SYMBOLS = r"[€£¥₹\$]"
_CURRENCY_SUFFIX_RE = re.compile(_NUMBER + r"\s?(" + _SYMBOLS + r")")
_CURRENCY_PREFIX_RE = re.compile(r"(" + _SYMBOLS + r")\s?" + _NUMBER)


def _amount_to_words(n: int, singular: str, plural: str) -> str:
    # French uses the singular below two ("zéro euro", "un euro").
    unit = singular if n < 2 else plural
    # "un million d'euros", "deux millions de dollars".
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{num2words(n, lang='fr')} {'d’' if unit[0] in 'aeiouy' else 'de '}{unit}"
    return f"{num2words(n, lang='fr')} {unit}"


def _currency_words(symbol: str, whole_str: str, frac_str: str | None) -> str:
    singular, plural, c_singular, c_plural = _CURRENCY_MAP[symbol]
    result = _amount_to_words(int(re.sub(r"\D", "", whole_str)), singular, plural)
    if frac_str:
        frac = int(frac_str[:2].ljust(2, "0"))
        if frac > 0:
            # "douze euros cinquante" is how an amount is read aloud; keep the subunit name
            # only for currencies without a common short form.
            result += f" {num2words(frac, lang='fr')}"
            if symbol != "€" and c_singular and c_plural:
                result += f" {c_singular if frac < 2 else c_plural}"
    return result


async def expand_currency(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand currency amounts to their spoken French form.

    Example::

        await expand_currency("Cela coûte 12,50 €", "*")
        # "Cela coûte douze euros cinquante"
    """
    text = _CURRENCY_SUFFIX_RE.sub(lambda m: _currency_words(m.group(3), m.group(1), m.group(2)), text)
    return _CURRENCY_PREFIX_RE.sub(lambda m: _currency_words(m.group(1), m.group(2), m.group(3)), text)


# --- Percentages -------------------------------------------------------------

_PERCENT_RE = re.compile(_NUMBER + r"\s?%")


async def expand_percentages(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand percentages to their spoken French form.

    Example::

        await expand_percentages("25 % de plus", "*")
        # "vingt-cinq pour cent de plus"
    """
    return _PERCENT_RE.sub(lambda m: f"{_number_words(m.group(1), m.group(2))} pour cent", text)


# --- Units -------------------------------------------------------------------

# (singular, plural). Keys are matched case-sensitively: French byte units (``Go``, ``Mo``) and
# SI prefixes (``mm`` vs ``Mm``) depend on case.
_UNIT_MAP: dict[str, tuple[str, str]] = {
    "km/h": ("kilomètre heure", "kilomètres heure"),
    "m/s": ("mètre par seconde", "mètres par seconde"),
    "km": ("kilomètre", "kilomètres"),
    "cm": ("centimètre", "centimètres"),
    "mm": ("millimètre", "millimètres"),
    "m": ("mètre", "mètres"),
    "kg": ("kilo", "kilos"),
    "mg": ("milligramme", "milligrammes"),
    "g": ("gramme", "grammes"),
    "cl": ("centilitre", "centilitres"),
    "ml": ("millilitre", "millilitres"),
    "l": ("litre", "litres"),
    "L": ("litre", "litres"),
    "°C": ("degré Celsius", "degrés Celsius"),
    "°F": ("degré Fahrenheit", "degrés Fahrenheit"),
    "°": ("degré", "degrés"),
    "To": ("téraoctet", "téraoctets"),
    "Go": ("gigaoctet", "gigaoctets"),
    "Mo": ("mégaoctet", "mégaoctets"),
    "Ko": ("kilooctet", "kilooctets"),
    "ko": ("kilooctet", "kilooctets"),
    "GHz": ("gigahertz", "gigahertz"),
    "MHz": ("mégahertz", "mégahertz"),
    "kHz": ("kilohertz", "kilohertz"),
    "Hz": ("hertz", "hertz"),
    "kW": ("kilowatt", "kilowatts"),
    "W": ("watt", "watts"),
    "V": ("volt", "volts"),
}

# Longest units first so ``km/h`` wins over ``km`` and ``°C`` over ``°``. Unit letters must not
# be followed by another letter ("5 minutes" is not "5 m" + "inutes").
_UNIT_RE = re.compile(
    _NUMBER + r"\s?(" + "|".join(re.escape(u) for u in sorted(_UNIT_MAP, key=len, reverse=True)) + r")(?![\w/])"
)


def _unit_match(match: re.Match) -> str:
    value = _to_number(match.group(1), match.group(2))
    singular, plural = _UNIT_MAP[match.group(3)]
    return f"{num2words(value, lang='fr')} {singular if value < 2 else plural}"


async def expand_units(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand unit abbreviations (and their quantity) to spoken French.

    Unlike Pipecat's English ``expand_units``, the number is verbalized too: the
    NeMo-Speech French TN grammar drops quantities like ``3,5 km``.

    Example::

        await expand_units("Il reste 3,5 km", "*")
        # "Il reste trois virgule cinq kilomètres"
    """
    return _UNIT_RE.sub(_unit_match, text)


# --- Dates -------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_FR_DATE_RE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")

_MONTH_NAMES = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


def _date_to_spoken(year: int, month: int, day: int) -> str | None:
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    day_words = "premier" if day == 1 else num2words(day, lang="fr")
    return f"{day_words} {_MONTH_NAMES[month - 1]} {num2words(year, lang='fr')}"


async def normalize_dates(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand ISO (``YYYY-MM-DD``) and French (``DD/MM/YYYY``) dates to spoken French.

    Example::

        await normalize_dates("Rendez-vous le 03/03/2024", "*")
        # "Rendez-vous le trois mars deux mille vingt-quatre"
    """
    text = _ISO_DATE_RE.sub(
        lambda m: _date_to_spoken(int(m.group(1)), int(m.group(2)), int(m.group(3))) or m.group(0), text
    )
    return _FR_DATE_RE.sub(
        lambda m: _date_to_spoken(int(m.group(3)), int(m.group(2)), int(m.group(1))) or m.group(0), text
    )


# --- Times -------------------------------------------------------------------

# "15h30", "15 h 30", "15h", "15:30" (a colon time needs two minute digits).
_TIME_H_RE = re.compile(r"\b([01]?\d|2[0-3])\s?h\s?([0-5]\d)?(?![\w])")
_TIME_COLON_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _time_to_spoken(hours: int, minutes: int) -> str:
    hour_words = "une" if hours == 1 else num2words(hours, lang="fr")
    # "vingt et une heures", not "vingt et un heures".
    if hours == 21:
        hour_words = "vingt et une"
    result = f"{hour_words} {'heure' if hours < 2 else 'heures'}"
    if minutes:
        result += f" {num2words(minutes, lang='fr')}"
    return result


async def expand_times(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand ``15h30`` / ``15:30`` style times to spoken French.

    Example::

        await expand_times("Rendez-vous à 15h30", "*")
        # "Rendez-vous à quinze heures trente"
    """
    text = _TIME_H_RE.sub(lambda m: _time_to_spoken(int(m.group(1)), int(m.group(2) or 0)), text)
    return _TIME_COLON_RE.sub(lambda m: _time_to_spoken(int(m.group(1)), int(m.group(2))), text)


# --- Phone numbers -----------------------------------------------------------

# "06 12 34 56 78", "06.12.34.56.78", "0612345678", "+33 6 12 34 56 78".
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+33[\s.\-]?|0)([1-9])((?:[\s.\-]?\d{2}){4})(?!\d)")


def _phone_match(match: re.Match) -> str:
    pairs = re.findall(r"\d{2}", match.group(2))
    # Read the French way, pair by pair: "zéro six douze trente-quatre ...".
    return " ".join([f"zéro {num2words(int(match.group(1)), lang='fr')}"] + [_pair_words(p) for p in pairs])


def _pair_words(pair: str) -> str:
    return f"zéro {num2words(int(pair[1]), lang='fr')}" if pair[0] == "0" else num2words(int(pair), lang="fr")


async def expand_phone_numbers(text: str, aggregation_type: str | AggregationType) -> str:
    """Expand French phone numbers to spoken digit pairs.

    Example::

        await expand_phone_numbers("Appelez le 06 12 34 56 78", "*")
        # "Appelez le zéro six douze trente-quatre cinquante-six soixante-dix-huit"
    """
    return _PHONE_RE.sub(_phone_match, text)


# --- Thousands separators ---------------------------------------------------

_GROUPED_NUMBER_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?![\d])")


async def join_thousands(text: str, aggregation_type: str | AggregationType) -> str:
    """Drop space / NBSP thousands separators so the number is read as a whole.

    Example::

        await join_thousands("Il y a 1 234 personnes", "*")
        # "Il y a 1234 personnes"
    """
    return _GROUPED_NUMBER_RE.sub(lambda m: re.sub(r"\D", "", m.group(0)), text)


# --- Email -------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _email_to_spoken(match: re.Match) -> str:
    local, domain = match.group(0).split("@", 1)
    local_spoken = (
        local.replace(".", " point ").replace("_", " tiret bas ").replace("-", " tiret ").replace("+", " plus ")
    )
    domain_spoken = domain.replace(".", " point ").replace("-", " tiret ")
    return f"{local_spoken} arobase {domain_spoken}"


async def email_to_speech(text: str, aggregation_type: str | AggregationType) -> str:
    """Transform email addresses into their spoken French form.

    Example::

        await email_to_speech("Écrivez à jean.dupont@example.fr", "*")
        # "Écrivez à jean point dupont arobase example point fr"
    """
    return _EMAIL_RE.sub(_email_to_spoken, text)
