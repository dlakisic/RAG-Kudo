import src.utils.validation as validation


def test_validate_api_keys_requires_openai_for_openai_provider(monkeypatch):
    monkeypatch.setattr(validation.settings, "llm_provider", "openai")
    monkeypatch.setattr(validation.settings, "openai_api_key", None)
    monkeypatch.setattr(validation.settings, "anthropic_api_key", "ak-test")
    monkeypatch.setattr(validation, "requires_openai_embeddings", lambda *_: False)

    assert validation.validate_api_keys() is False


def test_validate_api_keys_passes_for_anthropic_with_local_embeddings(monkeypatch):
    monkeypatch.setattr(validation.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(validation.settings, "openai_api_key", None)
    monkeypatch.setattr(validation.settings, "anthropic_api_key", "ak-test")
    monkeypatch.setattr(validation, "requires_openai_embeddings", lambda *_: False)

    assert validation.validate_api_keys() is True


def test_validate_api_keys_requires_openai_when_embeddings_are_openai(monkeypatch):
    monkeypatch.setattr(validation.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(validation.settings, "openai_api_key", None)
    monkeypatch.setattr(validation.settings, "anthropic_api_key", "ak-test")
    monkeypatch.setattr(validation, "requires_openai_embeddings", lambda *_: True)

    assert validation.validate_api_keys() is False
