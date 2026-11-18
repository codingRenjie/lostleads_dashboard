from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import PROJECT_ROOT, settings


class Base(DeclarativeBase):
    pass


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    losses: Mapped[list["Loss"]] = relationship(back_populates="batch", cascade="all, delete-orphan")
    file_errors: Mapped[list["FileError"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )
    insight: Mapped["BatchInsight | None"] = relationship(
        back_populates="batch", uselist=False, cascade="all, delete-orphan"
    )


class Loss(Base):
    __tablename__ = "losses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loss_id: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"), index=True)
    date: Mapped[str] = mapped_column(String(32))
    consultant: Mapped[str] = mapped_column(String(64))
    vehicle_model: Mapped[str] = mapped_column(String(256))
    customer_tags: Mapped[str] = mapped_column(Text, default="[]")
    loss_description: Mapped[str] = mapped_column(Text)
    source_filename: Mapped[str] = mapped_column(String(512))

    batch: Mapped["Batch"] = relationship(back_populates="losses")


class FileError(Base):
    __tablename__ = "file_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    message: Mapped[str] = mapped_column(Text)

    batch: Mapped["Batch"] = relationship(back_populates="file_errors")


class BatchInsight(Base):
    __tablename__ = "batch_insights"

    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"), primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text)

    batch: Mapped["Batch"] = relationship(back_populates="insight")


def _sqlite_engine_url() -> str:
    path = settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


engine = create_engine(
    _sqlite_engine_url(),
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
