"""Quote matching against real PDF-extracted text (P0: verifier false negatives).

The deterministic quote check is a *code* guarantee, so its failures are code
bugs. An audit of the archived runs found 29/131 quotes rejected by the
verifier, and nearly all of them are actually present in the paper: PDF
extraction breaks them in ways the old normalizer (collapse-whitespace +
lowercase) could not repair:

* ``English-\\nto-German``      - a hyphenated word split by a line break
* ``sur-\\nprisingly``          - the same, mid-word
* ``the sequence\\n6\\nlength``  - a page number injected into the sentence
* ``ensembles, by``            - punctuation the quote dropped
* ``the big model ... BLEU``   - an ellipsis-elided quote

These tests run against the *committed* example papers (real archives), so
they are regression tests on real failure modes rather than synthetic ones.
The reverse tests matter just as much: the fix must not make the verifier
accept quotes that are genuinely absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperflow.tools.texttools import verify_quote

REPO = Path(__file__).resolve().parents[1]
ATTENTION = (REPO / "examples/attention-is-all-you-need/artifacts/full_text.txt").read_text(encoding="utf-8")
WEREWOLF = (REPO / "examples/werewolf-dbn/artifacts/full_text.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The exact quotes the pre-fix verifier rejected although they are in the paper
# ---------------------------------------------------------------------------

REAL_FALSE_NEGATIVES = [
    # hyphen split across a line break ("English-\nto-German")
    "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, "
    "improving over the existing best results, including ensembles by over 2 BLEU",
    # line-break hyphen + the quote dropping an inline comma (", by over 2 BLEU.")
    "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, "
    "improving over the existing best results, including ensembles, by over 2 BLEU",
    # hyphen inside a wrapped word ("sequence-\naligned RNNs")
    "the Transformer is the first transduction model relying entirely on self-attention to "
    "compute representations of its input and output without using sequence-aligned RNNs "
    "or convolution",
    # mid-word line-break hyphen ("sur-\nprisingly well")
    "our model performs surprisingly well, yielding better results than all previously "
    "reported models with the exception of the Recurrent Neural Network Grammar",
    # page number "6" extracted into the middle of the sentence
    "self-attention layers are faster than recurrent layers when the sequence length n is "
    "smaller than the representation dimensionality d",
    # elided middle ("the big transformer model (Transformer (big) in Table 2)")
    "the big transformer model ... outperforms the best previously reported models "
    "(including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art "
    "BLEU score of 28.4",
]


@pytest.mark.parametrize("quote", REAL_FALSE_NEGATIVES, ids=range(len(REAL_FALSE_NEGATIVES)))
def test_archived_false_negatives_are_now_found(quote: str):
    result = verify_quote(ATTENTION, quote)
    assert result["found"] is True, f"{result['reason']}: {quote[:70]!r}"


def test_line_break_hyphen_is_repaired():
    text = "Our model achieves 28.4 BLEU on the WMT 2014 English-\nto-German task."
    result = verify_quote(text, "WMT 2014 English-to-German task")
    assert result["found"] is True
    assert result["match_mode"] != "verbatim"  # the repair is reported, not hidden


def test_soft_hyphen_and_zero_width_are_stripped():
    text = "the Trans\u00adformer\u200b achieves 28.4 BLEU"
    assert verify_quote(text, "the Transformer achieves 28.4 BLEU")["found"] is True


def test_page_number_noise_between_words_is_skipped():
    text = (
        "computational complexity, self-attention layers are faster than recurrent "
        "layers when the sequence\n6\nlength n is smaller than the representation "
        "dimensionality d, which is most often the case"
    )
    result = verify_quote(
        text,
        "self-attention layers are faster than recurrent layers when the sequence "
        "length n is smaller than the representation dimensionality d",
    )
    assert result["found"] is True
    assert result["match_mode"] == "token_aligned"


def test_inline_punctuation_difference_is_tolerated():
    text = "improving over the existing best results, including\nensembles, by over 2 BLEU."
    assert verify_quote(text, "including ensembles by over 2 BLEU")["found"] is True


def test_elided_quote_matches_segment_wise():
    text = "the big transformer model (Transformer (big)\nin Table 2) outperforms the best previously reported models"
    result = verify_quote(text, "the big transformer model ... outperforms the best previously reported models")
    assert result["found"] is True
    assert result["segments"] == 2


def test_match_mode_reports_verbatim_when_nothing_was_repaired():
    result = verify_quote("the win rate rises to 68.8% with dynamic beliefs", "win rate rises to 68.8%")
    assert result["match_mode"] == "verbatim"


# ---------------------------------------------------------------------------
# Reverse tests: the verifier must still reject quotes that are NOT in the text
# ---------------------------------------------------------------------------

def test_genuinely_absent_quote_still_fails_on_real_archive():
    """A real archived claim whose quote the Reader invented (never in the paper)."""
    result = verify_quote(WEREWOLF, "44.2% without beliefs and 68.8% with dynamic beliefs")
    assert result["found"] is False
    assert result["match_mode"] == "none"
    assert result["loc"] is None and result["context"] is None


def test_changed_number_still_fails():
    """A quote whose *number* is edited must never pass - numbers carry the claim."""
    assert verify_quote(ATTENTION, "we ran 300 games with 9 players")["found"] is False
    assert (
        verify_quote(
            ATTENTION,
            "our model establishes a new single-model state-of-the-art BLEU score of 44.8",
        )["found"]
        is False
    )


def test_reordered_wording_still_fails():
    assert (
        verify_quote(
            ATTENTION,
            "recurrent layers are faster than self-attention layers when the sequence length is larger",
        )["found"]
        is False
    )


def test_word_salad_spanning_the_whole_paper_still_fails():
    """Tokens borrowed from far-apart places must not be glued into a match."""
    salad = (
        "the Transformer encoder attention masks dropout label smoothing beam search "
        "positional encoding residual connections layer normalization"
    )
    assert verify_quote(ATTENTION, salad)["found"] is False


def test_short_quotes_are_still_rejected():
    assert verify_quote(ATTENTION, "the")["found"] is False
    assert verify_quote(ATTENTION, "BLEU")["found"] is False


def test_elided_quote_still_needs_every_segment():
    result = verify_quote(ATTENTION, "the big transformer model ... wins the Nobel prize in 2014")
    assert result["found"] is False
