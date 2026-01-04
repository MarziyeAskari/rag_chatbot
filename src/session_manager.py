import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class Message:
   role: str
   content: str
   timestamp: datetime = field(default_factory=datetime.now)
   metadata: Dict = field(default_factory=dict)

@dataclass
class Session:
    session_id: str
    created_at: datetime
    last_accessed: datetime
    messages: List[Message] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_message(self, role:str, content:str, metadata:Optional[Dict]):
        message= Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.last_accessed = datetime.now()

    def get_conversation_history(self, max_messages: Optional[int] = None) -> List[Tuple[str,str]]:

        messages_= self.messages
        if max_messages:
            messages_ = messages_[-max_messages:]
        return [(msg.role, msg.content) for msg in messages_]

    def get_conversation_context(self, max_messages: Optional[int] = None) -> str:
        history = self.get_conversation_history(max_messages)
        if not history:
            return ""
        context_parts= []
        for role, content in history:
            role_label="User" if role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {content}")
        return " ".join(context_parts)


class SessionManager:
    def __init__(self, session_timeout: int = 24, max_history_per_session:int = 50):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(hours=session_timeout)
        self.max_history_per_session = max_history_per_session
        self.lock = Lock()

    def create_session(self, session_id:Optional[str]=None, metadata: Optional[Dict]=None) -> Session:
        if session_id is None:
            session_id =str(uuid.uuid4())
        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, returning existing session")
                return self.sessions[session_id]
            session = Session(
                session_id=session_id,
              created_at=datetime.now(),
            last_accessed=datetime.now(),
            metadata=metadata or {}
            )
            self.sessions[session_id] = session
            logger.info(f"Created session {session_id}")
            return session

    def get_session(self, session_id:str) -> Optional[Session]:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            if datetime.now() - session.last_accessed > self.session_timeout:
                logger.info(f"Session {session_id} has expired,removing")
                del self.sessions[session_id]
                return None
            return session

    def get_or_create_session(self, session_id:Optional[str]) -> Session:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(session_id=session_id)

    def add_message(self, session_id:str , role:str, content:str, metadata:Optional[Dict]=None):
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} does not exist")
        Session.add_message(role, content, metadata)
        if len(session.messages) > self.max_history_per_session:
            with self.lock:
                session.messages = session.messages[-self.max_history_per_session:]

    def get_conversation_history(self, session_id:str, max_message:Optional[int]=None) -> List[Tuple[str,str]]:
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_conversation_history(max_message)
    def get_conversation_context(self, session_id:str, max_message:Optional[int]=10) -> str:
        session = self.get_session(session_id)
        if not session:
            return ""
        return session.get_conversation_context(max_message)
    def delete_session(self, session_id:str) -> bool:
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
            return False

    def cleanup_expired_sessions(self):
        with self.lock:
            expired= [
                session_id for session_id, session in self.sessions.items()
                if datetime.now()-session.last_accessed > self.session_timeout
            ]
            for session_id in expired:
                self.delete_session(session_id)
                logger.info(f"Cleanup expired session {session_id}")
            return len(expired)

    def get_session_count(self) -> int:
        with self.lock:
            return len(self.sessions)

    def get_all_sessions(self) -> List[Session]:
        with self.lock:
            return list(self.sessions.values())

