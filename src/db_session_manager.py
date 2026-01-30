
import uuid
from datetime import timedelta, datetime
from typing import Optional, Dict, Tuple, List
import logging
from src.database import DatabaseManager, SessionModel, MessageModel
from sqlalchemy.orm import Session as SQLSession

from src.session_manager import Session, Message

logger = logging.getLogger(__name__)


class DatabaseSessionManager:
    def __init__(
            self,
            database_url: str,
            database_path: str,
            session_timeout: int = 24,
            max_history_per_session: int = 50):
        self.db_manager = DatabaseManager(database_url=database_url, database_path=database_path)
        self.session_timeout = timedelta(hours=session_timeout)
        self.max_history_per_session = max_history_per_session

    def _get_db_session(self) -> SQLSession:
        return self.db_manager.get_session()

    def create_session(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Session:
        if session_id is None:
            session_id = str(uuid.uuid4())

        db = self._get_db_session()
        try:
            existing = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if existing:
                logger.warning(f"Session id {session_id} already exists")
                return self._load_session_from_db(existing, db)
            now = datetime.utcnow()
            session_model = SessionModel(
                session_id=session_id,
                created_at=now,
                last_accessed=now,
                metadata_json=metadata or {},
            )
            db.add(session_model)
            db.commit()
            db.refresh(session_model)

            logger.info(f"Created session id {session_id}")
            return  Session(
            session_id=session_model.session_id,
            created_at=session_model.created_at,
            last_accessed=session_model.last_accessed,
            messages=[],
            metadata=session_model.metadata_json or {},
        )
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create session id {session_id}: {str(e)}")
            raise
        finally:
            db.close()

    def _load_session_from_db(self, session_model: SessionModel, db: SQLSession):
        messages_models = db.query(MessageModel).filter(
            MessageModel.session_id == session_model.session_id).order_by(MessageModel.timestamp.desc()).all()
        messages = [
            Message(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                metadata=msg.metadata_json or {},
            )
            for msg in messages_models
        ]
        return Session(
            session_id=session_model.session_id,
            created_at=session_model.created_at,
            last_accessed=session_model.last_accessed,
            messages=messages,
            metadata=session_model.metadata_json or {},
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        db = self._get_db_session()
        try:
            session_model = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if session_model is None:
                return None
            if datetime.utcnow() - session_model.last_accessed > self.session_timeout:
                logger.info(f"Session id {session_id} has expired")
                return None
            return self._load_session_from_db(session_model, db)
        finally:
            db.close()

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(session_id)

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        db = self._get_db_session()
        try:
            session_model = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not session_model:
                raise ValueError(f"Session id {session_id} does not exist")

            message_model = MessageModel(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=datetime.utcnow(),
                metadata_json=metadata or {},
            )
            db.add(message_model)

            session_model.last_accessed = datetime.utcnow()

            # Trim history if too long
            message_count = db.query(MessageModel).filter(MessageModel.session_id == session_id).count()
            if message_count > self.max_history_per_session:
                to_delete = (
                    db.query(MessageModel)
                    .filter(MessageModel.session_id == session_id)
                    .order_by(MessageModel.timestamp.asc())
                    .limit(message_count - self.max_history_per_session)
                    .all()
                )
                for msg in to_delete:
                    db.delete(msg)

            db.commit()
            logger.debug(f"Added {role} message to session {session_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add message to session {session_id}: {str(e)}")
            raise
        finally:
            db.close()

    def get_conversation_history(self, session_id: str, max_messages: Optional[int] = None) -> List[Tuple[str, str]]:
        db = self._get_db_session()
        try:
            query = db.query(MessageModel).filter(
                MessageModel.session_id == session_id
            ).order_by(MessageModel.timestamp.asc())
            if max_messages:
                total = query.count()
                if total > max_messages:
                    query = query.offset(total - max_messages)
            messages = query.all()
            return [(msg.role, msg.content) for msg in messages]
        finally:
            db.close()

    def get_conversation_context(self, session_id: str, max_messages: Optional[int] = 10) -> str:
        history = self.get_conversation_history(session_id, max_messages=max_messages)
        if not history:
            return ""
        parts = []
        for role, content in history:
            role_label = "User" if role == "user" else "Assistant"
            parts.append(f"{role_label}: {content}")
        return "\n".join(parts)

    def delete_session(self, session_id: str) -> bool:
        db = self._get_db_session()
        try:
            session_model = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not session_model:
                return False
            db.query(MessageModel).filter(MessageModel.session_id == session_id).delete()
            db.delete(session_model)
            db.commit()
            logger.info(f"Deleted session id {session_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete session id {session_id}: {str(e)}")
            raise
        finally:
            db.close()

    def cleanup_expired_sessions(self) -> int:
        db = self._get_db_session()
        try:
            cutoff_time = datetime.utcnow() - self.session_timeout
            expired_sessions = db.query(SessionModel).filter(
                SessionModel.last_accessed < cutoff_time
            ).all()
            expired_count = 0
            for session in expired_sessions:
                db.query(MessageModel).filter(MessageModel.session_id == session.session_id).delete()
                expired_count += 1
                db.delete(session)
                logger.info(f"Deleted session id {session.session_id}")
            db.commit()
            return expired_count
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete expired sessions: {str(e)}")
            raise
        finally:
            db.close()

    def get_all_sessions(self) -> List[Session]:
        db = self._get_db_session()
        try:
            cutoff_time = datetime.utcnow() - self.session_timeout
            session_models = db.query(SessionModel).filter(
                SessionModel.last_accessed < cutoff_time
            ).all()
            return [self._load_session_from_db(session_model, db) for session_model in session_models]
        finally:
            db.close()

    def close(self):
        self.db_manager.close()
