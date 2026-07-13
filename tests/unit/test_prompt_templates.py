import pytest
from cap.data.prompt_templates import PromptTemplate, PROMPT_TYPES, FACTUALITY, RAW


def test_apply_factuality():
    result = FACTUALITY.apply("The sky is green.")
    assert "The sky is green." in result
    assert "true or false" in result


def test_apply_raw():
    text = "hello world"
    assert RAW.apply(text) == text


def test_constructor_requires_text_placeholder():
    with pytest.raises(ValueError):
        PromptTemplate("no placeholder here")


def test_all_prompt_types_present():
    for key in ["factuality", "faithfulness", "negation", "raw"]:
        assert key in PROMPT_TYPES


def test_from_dict():
    t = PromptTemplate.from_dict({"template": "Is {text} correct? Answer:"})
    assert t.apply("the sky is blue") == "Is the sky is blue correct? Answer:"


def test_repr_roundtrips_template():
    t = PromptTemplate("Q: {text}")
    assert repr(t) == "PromptTemplate('Q: {text}')"
