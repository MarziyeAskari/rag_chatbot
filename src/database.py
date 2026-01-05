from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy import Column, Integer, DateTime, JSON, Index, ForeignKey, String, create_engine, StaticPool
from sqlalchemy.orm import declarative_base, sessionmaker,Session as DBSession
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()
class SessionModel(Base):
    __tablename__ = "session"
    session_id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata_json = Column(JSON, nullable=False)

    __table_args__ = (
        Index("idx_last_accessed", last_accessed),
    )

    def to_dict(self) ->Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata_json": self.metadata_json or {}
        }


class MessageModel(Base):
    __tablename__ = "message"
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("session.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow ,nullable=False, index=True)
    metadata_json = Column(JSON, default=dict)

    __table_args__ = (
        Index("idx_session_timestamp", timestamp),
    )
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp if self.timestamp else None,
            "metadata_json": self.metadata_json or {}
        }

class DatabaseManager:
    def __init__(self, database_url:Optional[str]=None,databae_path:Optional[str]=None):
        if database_url is None:
            self.database_url = database_url
        elif databae_path:
            db_path = Path(databae_path)
            db_path = db_path.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{db_path}"
        else:
            default_path = Path("./data/sessions.db")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{default_path}"

        if self.database_url.startswith("sqlite://"):
            self.engine=create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            self.engine = create_engine(self.database_url,
                                        pool_pre_ping=True,
                                        echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
        logger.info(f"Database created at {self.database_url}")

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)
        logger.info("Tables created")
    def get_session(self) -> DBSession:
        return self.SessionLocal()
    def close(self):
        self.engine.dispose()
        logger.info(f"Database closed at {self.database_url}")

