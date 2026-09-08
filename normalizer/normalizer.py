# Copyright 2022 The OpenAI team and The HuggingFace Team. All rights reserved.
# Most of the code is copy pasted from the original whisper repository
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
from .english_abbreviations import english_name_normalizer, english_spelling_normalizer, english_compound_normalizer

import regex


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


def remove_symbols_keep_marks(s: str):
    """
    Replace symbols and punctuation with a space, keeping combining marks.

    Unlike `remove_symbols`, combining marks (category 'M') are preserved. This
    is required for scripts like Devanagari, where vowel signs (matras) and the
    virama are combining marks that are integral to words.
    """
    return "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFKC", s))


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s


class BasicMultilingualTextNormalizer:
    def __init__(self, remove_diacritics: bool = True):
        # When keeping diacritics, also keep combining marks (category 'M'):
        # scripts like Devanagari encode vowel signs and the virama as
        # combining marks, so stripping them mangles words.
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols_keep_marks

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        # Remove punctuation and extra spaces
        s = regex.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        return s


class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"o", "oh", "zero"}
        # fmt: off
        self.ones = {
            name: i
            for i, name in enumerate(
                ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"],
                start=1,
            )
        }
        # fmt: on
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s") for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()}
        self.tens_ordinal = {name.replace("y", "ieth"): (value, "th") for name, value in self.tens.items()}
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {name + "s": (value, "s") for name, value in self.multipliers.items()}
        self.multipliers_ordinal = {name + "th": (value, "th") for name, value in self.multipliers.items()}
        self.multipliers_suffixed = {**self.multipliers_plural, **self.multipliers_ordinal}
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ]
            for key in mapping
        }
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def is_digit_token(token: Optional[str]) -> bool:
            """True for tokens that continue a digit sequence ("four", "oh", "20")."""
            return token is not None and bool(
                re.match(r"^\d+$", token)
                or token in self.zeros
                or token in self.ones
                or token in self.tens
            )

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for i, current in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                # "oh" is far more often the interjection than a spoken zero, so
                # read it as a digit only inside a digit sequence: the sequence
                # has to continue on the right ("four oh one", "nineteen oh
                # five"), and with nothing pending on the left it takes two more
                # digit tokens, i.e. a serial/phone-style reading ("oh seven nine
                # eight"). On its own — including the doubled "oh oh" — it stays
                # a word. "o" and "zero" are unchanged.
                next2 = words[i + 2] if i + 2 < len(words) else None
                in_number = is_digit_token(next) and (
                    value is not None or is_digit_token(next2)
                )
                if current == "oh" and not in_number:
                    if value is not None:
                        yield output(value)  # don't drop a pending number
                    yield output(current)
                else:
                    value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:  # replace the last zero with the digit
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # replace "<number> and a half" with "<number> point five"
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # write "one(s)" instead of "1(s)", just for the readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self, english_spelling_mapping):
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishAcronymNormalizer:
    """
    Collapse sequences of single-character tokens (letters or digits) into single words.

    This normalizes acronym spacing so that both spaced-out and joined forms match:
        - "b b c" -> "bbc"
        - "5 g" -> "5g"

    Lone single-character words surrounded by multi-character words are left untouched
    (e.g. "a big cat" stays "a big cat").
    """

    def __call__(self, s: str) -> str:
        words = s.split()
        result = []
        i = 0
        while i < len(words):
            if len(words[i]) == 1 and words[i].isalnum():
                # Start of a potential acronym run
                run = [words[i]]
                j = i + 1
                while j < len(words) and len(words[j]) == 1 and words[j].isalnum():
                    run.append(words[j])
                    j += 1
                # Require 3+ tokens if the run contains common words "a" or "i",
                # otherwise 2+ is enough (e.g. "5 g" -> "5g")
                has_common_word = any(c in ("a", "i") for c in run)
                min_run = 3 if has_common_word else 2
                if len(run) >= min_run:
                    result.append("".join(run))
                else:
                    result.extend(run)
                i = j
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)


class EnglishNameNormalizer:
    """
    Collapse common name spelling variants to a single canonical form.

    This is intentionally conservative and token-based so it can be extended
    with project-specific aliases when needed.
    """

    def __init__(self, english_name_mapping=english_name_normalizer):
        self.mapping = english_name_mapping

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTimeNormalizer:
    """Canonicalize explicit clock expressions before general number normalization."""

    def __init__(self, number_normalizer):
        self.number_normalizer = number_normalizer
        digits = "|".join(word for word, value in number_normalizer.ones.items() if value < 10)
        small_numbers = "|".join(number_normalizer.ones)
        number = rf"(?:[0-9]{{1,2}}|(?:twenty|thirty|forty|fifty)(?:[\s-]+(?:{digits}))?|{small_numbers}|zero)"
        clock = r"o\s*['\u2019]?\s*clock"
        self.relative_time = re.compile(
            rf"(?<![\w.:])(?P<amount>half|(?:a\s+)?quarter|{number}\s+minutes?)"
            rf"\s+(?P<relation>past|after|to|before)\s+(?P<hour>{number})"
            rf"(?:\s+{clock})?\b"
        )
        self.on_the_hour = re.compile(rf"(?<![\w.:])(?P<hour>{number})\s+{clock}\b")
        self.numeric_time = re.compile(
            r"(?<![\w:./])(?P<hour>[01]?[0-9]|2[0-3]):(?P<minute>[0-5][0-9])(?![\w:]|\.[0-9])"
        )

    def _number(self, text):
        value = self.number_normalizer(text.replace("-", " ")).strip()
        return 1 if value == "one" else int(value)

    def _relative_time(self, match):
        hour = self._number(match["hour"])
        if not 0 <= hour <= 23:
            return match.group()
        amount = match["amount"]
        if amount == "half":
            minute = 30
        elif amount.endswith("quarter"):
            minute = 15
        else:
            minute = self._number(re.sub(r"\s+minutes?$", "", amount))
        if not 0 <= minute < 60:
            return match.group()
        if match["relation"] in ("to", "before") and minute:
            hour = (hour - 2) % 12 + 1 if 1 <= hour <= 12 else (hour - 1) % 24
            minute = 60 - minute
        return f"{hour}:{minute:02d}"

    def _on_the_hour(self, match):
        hour = self._number(match["hour"])
        return f"{hour}:00" if 0 <= hour <= 23 else match.group()

    def _numeric_time(self, match):
        hour, minute = int(match["hour"]), int(match["minute"])
        return str(hour) if minute == 0 else f"{hour}:{minute:02d}"

    def __call__(self, text):
        text = self.relative_time.sub(self._relative_time, text)
        text = self.on_the_hour.sub(self._on_the_hour, text)
        return self.numeric_time.sub(self._numeric_time, text)


class EnglishTextNormalizer:
    def __init__(self, english_spelling_mapping=english_spelling_normalizer):
        # Filler words / hesitations to remove. Written as regexes so that
        # arbitrary elongation is covered without enumerating every spelling
        # ("uh", "uhh", "uuuh", "uhhhh", ...). Each alternative is wrapped in
        # \b...\b below, so a shorter alternative cannot match a prefix of a
        # longer token and the order of the single-token patterns is irrelevant.
        # Hyphens are word boundaries too, which is why most hyphenated forms
        # need no entry ("um-hmm" is matched as "um" + "hmm") — but any whose
        # halves are not both fillers must be listed *before* the patterns,
        # otherwise only the first half is matched ("ah-ha" -> "ha").
        filler_words = [
            "ah-ha",         # "ha" alone is not a filler, so match the pair first
            r"a+h+m*",       # ah, aah, ahh, ahhh, aaah, ahm, ahmm
            r"a+h+a+",       # aha, ahaa, ahaaa
            r"e+h+m*",       # eh, ehh, eeeh, ehhh, ehm, ehmm
            r"e+m+",         # em, emm
            r"e+r+m*",       # er, err, errr, erm
            r"h+a+h+",       # hah, hahh
            r"h+e+h+",       # heh, hehh
            r"h+m+",         # hm, hmm, hmmm, hhm
            r"h+u+h+",       # huh, huhh
            r"m{2,}",        # mm, mmm, mmmm
            r"m+h+m*",       # mh, mhm, mhmm, mmhm
            r"t+s+k+",       # tsk
            r"u+g+h+",       # ugh, uuugh
            r"u+h+m*",       # uh, uuh, uhh, uhhh, uuuh, uhm, uuuhm
            r"u+h+u+[hm]*",  # uhuh, uhum
            r"u+m+h*",       # um, umm, ummm, uuum, umh
            # Irregular forms, not worth a pattern of their own.
            "ahem", "eheh", "ehehe", "ehr", "hmmph", "hum", "hunh", "mhum", "mmkay",
        ]
        self.ignore_patterns = r"\b(" + "|".join(filler_words) + r")\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"\b(it|he|she|what|that|who|here|there|how|when|where|why|this)'s\b": r"\1 is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_times = EnglishTimeNormalizer(self.standardize_numbers)
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)
        self.standardize_names = EnglishNameNormalizer()
        self.standardize_acronyms = EnglishAcronymNormalizer()
        # Multi-word compound mappings — defined in english_abbreviations.py
        self.compound_words = english_compound_normalizer

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe
        s = self.standardize_times(s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.(?![0-9])", " ", s)
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep some symbols for numerics

        # Normalize hardcoded compound words (e.g. "wi fi" -> "wifi" after hyphen removal)
        for pattern, replacement in self.compound_words.items():
            s = re.sub(pattern, replacement, s)

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)
        s = self.standardize_names(s)
        s = self.standardize_acronyms(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£](?![0-9])", " ", s)
        s = re.sub(r"(?<![0-9])%", " ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s
