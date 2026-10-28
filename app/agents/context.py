"""Trusted, server-injected context for one agent run."""
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """Identity and retrieval settings the model is never allowed to override.

    Tools read ``user_id`` and ``knowledge_base_id`` from here, never from the
    model-generated arguments, so the model can only choose *which* order to
    query, not *whose* order it is.
    """

    user_id: int
    session_id: int
    trace_id: str
    run_id: str = ""
    assistant_id: int | None = None
    knowledge_base_id: int | None = None
    locale: str = "zh-CN"
    permissions: set[str] = field(default_factory=set)
    retrieval_config: dict = field(default_factory=dict)

    def has_permission(self, scope: str) -> bool:
        """Return True when the user has the requested ``resource.action``."""
        return "*" in self.permissions or scope in self.permissions
