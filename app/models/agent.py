"""
Agent runtime persistence models.

``AgentRun.state_json`` is the source of truth for a paused run.  When a write
intent creates a PendingAction, the SSE stream ends; the conversation is later
resumed from this snapshot after the user confirms or cancels.
"""
from datetime import datetime

from app.extensions import db


class AgentRun(db.Model):
    """One user request handled by the agent loop."""

    __tablename__ = "agent_runs"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    trace_id = db.Column(db.String(64), nullable=False, index=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False, index=True)
    assistant_id = db.Column(db.Integer, db.ForeignKey("assistants.id"))

    status = db.Column(db.String(32), nullable=False, default="running")
    # running / awaiting_confirmation / completed / cancelled / failed
    step_count = db.Column(db.Integer, nullable=False, default=0)
    input = db.Column(db.Text, nullable=False, default="")

    state_json = db.Column(db.JSON, nullable=False, default=dict)

    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    steps = db.relationship(
        "AgentStep",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AgentStep.id",
    )

    def __repr__(self):
        return f"<AgentRun {self.run_id} {self.status}>"


class AgentStep(db.Model):
    """Audit row for one tool invocation."""

    __tablename__ = "agent_steps"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(
        db.String(64),
        db.ForeignKey("agent_runs.run_id"),
        nullable=False,
        index=True,
    )

    step_index = db.Column(db.Integer, nullable=False, default=0)
    tool_name = db.Column(db.String(64), nullable=False)
    tool_call_id = db.Column(db.String(128))
    args_json = db.Column(db.JSON, nullable=False, default=dict)
    result_code = db.Column(db.String(64))
    ok = db.Column(db.Boolean, nullable=False, default=False)
    duration_ms = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    run = db.relationship("AgentRun", back_populates="steps")

    def __repr__(self):
        return f"<AgentStep {self.tool_name} ok={self.ok}>"


class PendingAction(db.Model):
    """A write intent waiting for explicit user confirmation.

    The model never decides to execute a write.  It can only create one of
    these rows; the confirm endpoint performs the real transaction.
    """

    __tablename__ = "pending_actions"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(64), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False, index=True)

    action_type = db.Column(db.String(32), nullable=False)  # return / exchange
    resource_type = db.Column(db.String(32), nullable=False)
    resource_id = db.Column(db.Integer)

    payload_json = db.Column(db.JSON, nullable=False, default=dict)
    display_json = db.Column(db.JSON, nullable=False, default=dict)
    args_hash = db.Column(db.String(64), nullable=False)

    status = db.Column(db.String(32), nullable=False, default="pending")
    # pending / executing / success / failed / cancelled / expired

    idempotency_key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    tool_call_id = db.Column(db.String(128))

    expires_at = db.Column(db.DateTime, nullable=False)
    confirmed_at = db.Column(db.DateTime)
    executed_at = db.Column(db.DateTime)
    result_json = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<PendingAction {self.action_type} {self.status}>"
