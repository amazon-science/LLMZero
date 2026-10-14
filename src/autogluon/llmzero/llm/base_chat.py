import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def extract_text_content(content: Any) -> str:
    """Normalize an LLM response's ``content`` to a plain text string.

    With extended thinking enabled, Anthropic-on-Bedrock returns ``content`` as a
    list of blocks, e.g. ``[{"type": "thinking", ...}, {"type": "text", "text": ...}]``.
    Downstream agents expect a plain string, so drop the thinking/redacted blocks
    and concatenate the ``text`` blocks. A plain-string content is returned as-is.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # Keep only the visible answer text; skip thinking/redacted_thinking.
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif "text" in block and isinstance(block["text"], str) and "type" not in block:
                    texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts)

    return str(content) if content is not None else ""


def log_retry_attempt(retry_state):
    """Custom callback to log each retry attempt"""
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number
        logger.error(f"Attempt {attempt_number} failed: {type(exception).__name__}: {exception}")


class GlobalTokenTracker:
    """Singleton class to track token usage across all conversations (thread-safe)."""

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(GlobalTokenTracker, cls).__new__(cls)
                    cls._instance._lock = threading.Lock()
                    cls._instance.total_input_tokens = 0
                    cls._instance.total_output_tokens = 0
                    cls._instance.conversations = {}
                    cls._instance.sessions = {}
        return cls._instance

    def add_tokens(
        self,
        conversation_id: str,
        session_name: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Add token counts for a specific conversation and session."""
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            self.conversations[conversation_id]["input_tokens"] += input_tokens
            self.conversations[conversation_id]["output_tokens"] += output_tokens

            if session_name not in self.sessions:
                self.sessions[session_name] = {"input_tokens": 0, "output_tokens": 0}

            self.sessions[session_name]["input_tokens"] += input_tokens
            self.sessions[session_name]["output_tokens"] += output_tokens

    def get_conversation_usage(self, conversation_id: str) -> Dict[str, Any]:
        """Get token usage for a specific conversation."""
        with self._lock:
            if conversation_id not in self.conversations:
                return {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

            conv_usage = self.conversations[conversation_id]
            return {
                "input_tokens": conv_usage["input_tokens"],
                "output_tokens": conv_usage["output_tokens"],
                "total_conversation_tokens": conv_usage["input_tokens"] + conv_usage["output_tokens"],
            }

    def get_total_usage(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Get total token usage across all conversations and sessions."""
        with self._lock:
            usage_data = {
                "total": {
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_input_tokens + self.total_output_tokens,
                },
                "conversations": {},
                "sessions": {},
            }

            for conv_id, conv_usage in self.conversations.items():
                usage_data["conversations"][conv_id] = {
                    "input_tokens": conv_usage["input_tokens"],
                    "output_tokens": conv_usage["output_tokens"],
                    "total_tokens": conv_usage["input_tokens"] + conv_usage["output_tokens"],
                }

            for session_name, session_usage in self.sessions.items():
                usage_data["sessions"][session_name] = {
                    "input_tokens": session_usage["input_tokens"],
                    "output_tokens": session_usage["output_tokens"],
                    "total_tokens": session_usage["input_tokens"] + session_usage["output_tokens"],
                }

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(usage_data, f, indent=2)

        return usage_data


class BaseAssistantChat(BaseModel):
    """Base class for assistant chat models with LangGraph support."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens_: int = Field(default=0)
    output_tokens_: int = Field(default=0)
    graph: Optional[Any] = Field(default=None, exclude=True)
    app: Optional[Any] = Field(default=None, exclude=True)
    memory: Optional[Any] = Field(default=None, exclude=True)
    token_tracker: GlobalTokenTracker = Field(default_factory=GlobalTokenTracker)
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_name: str = Field(default="default_session")

    def initialize_conversation(
        self,
        llm: Any,
        system_prompt: str = "",
    ) -> None:
        """Initialize conversation using LangGraph."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        graph = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            prompt_messages = prompt_template.invoke(state)
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}

        graph.add_edge(START, "model")
        graph.add_node("model", call_model)

        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)

        self.graph = graph
        self.app = app
        self.memory = memory

    def describe(self) -> Dict[str, Any]:
        """Get model description and conversation history."""
        conversation_usage = self.token_tracker.get_conversation_usage(self.conversation_id)
        total_usage = self.token_tracker.get_total_usage()

        return {
            "history": self.history_,
            "conversation_tokens": conversation_usage,
            "total_tokens_across_all_conversations": total_usage,
            "session_name": self.session_name,
        }

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=32, min=32, max=128), after=log_retry_attempt)
    def assistant_chat(self, message) -> str:
        """Send a message and get response using LangGraph.

        Args:
            message: Either a string (text-only) or a dict with multimodal content
                    Dict format: {"multimodal": True, "content": [...]}

        Returns:
            The text response from the LLM
        """
        if not self.app:
            raise RuntimeError("Conversation not initialized. Call initialize_conversation first.")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Handle multimodal content
        if isinstance(message, dict) and message.get("multimodal"):
            # For multimodal content, pass the content blocks directly
            input_messages = [HumanMessage(content=message["content"])]
            # Store a text representation for history
            message_for_history = "[Multimodal content with images]"
        else:
            # Regular text message
            input_messages = [HumanMessage(content=message)]
            message_for_history = message

        response = self.app.invoke({"messages": input_messages}, config)

        ai_message = response["messages"][-1]
        input_tokens = output_tokens = 0

        if hasattr(ai_message, "usage_metadata"):
            usage = ai_message.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Update both instance and global tracking
            self.input_tokens_ += input_tokens
            self.output_tokens_ += output_tokens
            self.token_tracker.add_tokens(self.conversation_id, self.session_name, input_tokens, output_tokens)

        # With extended thinking, content is a list of blocks (thinking + text);
        # extract the visible text so downstream string parsers keep working.
        text_output = extract_text_content(ai_message.content)

        self.history_.append(
            {
                "input": message_for_history,
                "output": text_output,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        return text_output

    async def astream(self, message: str):
        """Stream responses using LangGraph."""
        if not self.app:
            raise RuntimeError("Conversation not initialized. Call initialize_conversation first.")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]

        async for chunk, metadata in self.app.stream({"messages": input_messages}, config, stream_mode="messages"):
            if isinstance(chunk, AIMessage):
                text = extract_text_content(chunk.content)
                if text:
                    yield text
