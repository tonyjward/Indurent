"""A scripted chat model for tests: no Claude calls, no API key.

It replies with the next message in its script each time it's called, so a test lays out the tool calls and the
final answer, and LangGraph runs the real tools in between. Used by the graph, chat and app tests.
"""

import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class Scripted(BaseChatModel):
    """A fake chat model that replies with the next scripted message each time it's called."""

    replies: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name, content="", **args):
    """A scripted tool call (with a unique id, as Claude gives), optionally with some text said before it."""
    return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": f"call-{uuid.uuid4().hex[:8]}"}])
