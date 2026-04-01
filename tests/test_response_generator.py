import asyncio
from types import SimpleNamespace

import src.generation.response_generator as rg_module
from src.generation.response_generator import KudoResponseGenerator


class FakeLangfuseManager:
    def is_enabled(self) -> bool:
        return False


class FakeLLMManager:
    def __init__(self):
        self.last_messages = None

    def chat(self, messages):
        self.last_messages = messages
        return "reponse-test"

    def stream_chat(self, messages):
        self.last_messages = messages
        yield "tok1"
        yield "tok2"


class FakeRetriever:
    def __init__(self, nodes):
        self.nodes = nodes
        self.last_retrieve = None
        self.last_retrieve_with_context = None

    def retrieve(self, query):
        self.last_retrieve = query
        return self.nodes

    def retrieve_with_context(self, query, context):
        self.last_retrieve_with_context = (query, context)
        return self.nodes


def _make_node(text: str, score: float = 0.8):
    inner = SimpleNamespace(
        metadata={
            "file_name": "reglement.pdf",
            "section": "Section A",
            "category": "scoring",
            "article_reference": "Article 3",
        },
        get_content=lambda: text,
    )
    return SimpleNamespace(node=inner, score=score)


def test_generate_uses_retriever_nodes_and_builds_sources(monkeypatch):
    monkeypatch.setattr(rg_module, "get_langfuse_manager", lambda: FakeLangfuseManager())

    nodes = [_make_node("Texte de regle")]
    llm = FakeLLMManager()
    retriever = FakeRetriever(nodes)
    generator = KudoResponseGenerator(index=object(), llm_manager=llm, retriever=retriever)

    result = generator.generate("Quelle regle s'applique ?")

    assert retriever.last_retrieve == "Quelle regle s'applique ?"
    assert result["answer"] == "reponse-test"
    assert result["num_sources"] == 1
    assert result["sources"][0]["article_reference"] == "Article 3"
    assert "CONTEXTE DU RÈGLEMENT OFFICIEL" in llm.last_messages[-1].content
    assert "Texte de regle" in llm.last_messages[-1].content


def test_generate_can_use_pre_retrieved_nodes(monkeypatch):
    monkeypatch.setattr(rg_module, "get_langfuse_manager", lambda: FakeLangfuseManager())

    nodes = [_make_node("Contexte externe")]
    llm = FakeLLMManager()
    retriever = FakeRetriever(nodes=[])
    generator = KudoResponseGenerator(index=object(), llm_manager=llm, retriever=retriever)

    result = generator.generate("Question", retrieved_nodes=nodes)

    assert retriever.last_retrieve is None
    assert result["num_sources"] == 1
    assert "Contexte externe" in llm.last_messages[-1].content


def test_generate_stream_yields_tokens_and_nodes(monkeypatch):
    monkeypatch.setattr(rg_module, "get_langfuse_manager", lambda: FakeLangfuseManager())

    nodes = [_make_node("Bloc contexte", score=0.2)]
    llm = FakeLLMManager()
    retriever = FakeRetriever(nodes)
    generator = KudoResponseGenerator(index=object(), llm_manager=llm, retriever=retriever)

    async def _collect():
        out = []
        async for token, streamed_nodes in generator.generate_stream("Question stream"):
            out.append((token, streamed_nodes))
        return out

    events = asyncio.run(_collect())

    assert [tok for tok, _ in events] == ["tok1", "tok2"]
    assert events[0][1] == nodes
    assert retriever.last_retrieve == "Question stream"
