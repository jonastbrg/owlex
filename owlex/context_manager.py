"""
ContextManager - SQLite-backed shared context for multi-agent sessions.

Provides persistent shared context that agents can read/write via MCP tools,
reducing token waste from duplicating context across agent calls.

Features:
- SQLite with WAL mode for concurrent access
- Optimistic concurrency via version/etag
- Namespace support for worktree/task isolation
- LRU + size quotas (500MB max, 7 day TTL)
- Auto-extend TTL on access
"""

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


# Default quotas
DEFAULT_MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500MB
DEFAULT_MAX_AGE_DAYS = 7
DEFAULT_DB_PATH = ".owlex/contexts.db"


@dataclass
class Context:
    """A shared context entry."""
    id: str
    namespace: str
    data: dict[str, Any]
    version: int
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    expires_at: datetime | None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "data": self.data,
            "version": self.version,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class ContextSummary:
    """Summary of a context (without full data)."""
    id: str
    namespace: str
    version: int
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    expires_at: datetime | None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "version": self.version,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class ContextManagerConfig:
    """Configuration for ContextManager."""
    db_path: str | Path = DEFAULT_DB_PATH
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES
    max_age_days: int = DEFAULT_MAX_AGE_DAYS
    working_directory: str | None = None

    def get_db_path(self) -> Path:
        """Get absolute path to database file."""
        if self.working_directory:
            base = Path(self.working_directory)
        else:
            base = Path.cwd()
        return base / self.db_path


class VersionConflictError(Exception):
    """Raised when optimistic concurrency check fails."""
    def __init__(self, context_id: str, expected_version: int, actual_version: int):
        self.context_id = context_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Version conflict for context '{context_id}': "
            f"expected version {expected_version}, but found {actual_version}"
        )


class ContextNotFoundError(Exception):
    """Raised when a context is not found."""
    def __init__(self, context_id: str):
        self.context_id = context_id
        super().__init__(f"Context '{context_id}' not found")


class ContextManager:
    """
    SQLite-backed shared context manager.

    Thread-safe with connection pooling via thread-local storage.
    Uses WAL mode for concurrent read/write access.
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS contexts (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            data BLOB NOT NULL,
            version INTEGER DEFAULT 1,
            size_bytes INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            expires_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_contexts_namespace ON contexts(namespace);
        CREATE INDEX IF NOT EXISTS idx_contexts_expires_at ON contexts(expires_at);
        CREATE INDEX IF NOT EXISTS idx_contexts_accessed_at ON contexts(accessed_at);
    """

    def __init__(self, config: ContextManagerConfig | None = None):
        """
        Initialize ContextManager.

        Args:
            config: Configuration options. If None, uses defaults.
        """
        self.config = config or ContextManagerConfig()
        self._local = threading.local()
        self._initialized = False
        self._init_lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            db_path = self.config.get_db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row

            self._local.connection = conn

        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize(self):
        """Initialize the database schema."""
        with self._init_lock:
            if self._initialized:
                return

            with self._transaction() as conn:
                conn.executescript(self._CREATE_TABLE_SQL)

            self._initialized = True

    def _ensure_initialized(self):
        """Ensure database is initialized."""
        if not self._initialized:
            self.initialize()

    def _now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)

    def _parse_timestamp(self, ts: str | None) -> datetime | None:
        """Parse ISO timestamp string."""
        if ts is None:
            return None
        return datetime.fromisoformat(ts)

    def _generate_id(self) -> str:
        """Generate a unique context ID."""
        return f"ctx-{uuid.uuid4().hex[:12]}"

    def _calculate_expiry(self) -> datetime:
        """Calculate expiry time based on max_age_days."""
        return self._now() + timedelta(days=self.config.max_age_days)

    def _row_to_context(self, row: sqlite3.Row) -> Context:
        """Convert database row to Context object."""
        return Context(
            id=row["id"],
            namespace=row["namespace"],
            data=json.loads(row["data"]),
            version=row["version"],
            size_bytes=row["size_bytes"],
            created_at=self._parse_timestamp(row["created_at"]),
            updated_at=self._parse_timestamp(row["updated_at"]),
            accessed_at=self._parse_timestamp(row["accessed_at"]),
            expires_at=self._parse_timestamp(row["expires_at"]),
        )

    def _row_to_summary(self, row: sqlite3.Row) -> ContextSummary:
        """Convert database row to ContextSummary object."""
        return ContextSummary(
            id=row["id"],
            namespace=row["namespace"],
            version=row["version"],
            size_bytes=row["size_bytes"],
            created_at=self._parse_timestamp(row["created_at"]),
            updated_at=self._parse_timestamp(row["updated_at"]),
            accessed_at=self._parse_timestamp(row["accessed_at"]),
            expires_at=self._parse_timestamp(row["expires_at"]),
        )

    def create(
        self,
        namespace: str,
        data: dict[str, Any],
        ttl_days: int | None = None,
    ) -> Context:
        """
        Create a new context.

        Args:
            namespace: Namespace for isolation (e.g., "task:123", "worktree:feature-x")
            data: Context data to store
            ttl_days: Time-to-live in days. If None, uses config default.

        Returns:
            Created Context with id and version
        """
        self._ensure_initialized()

        context_id = self._generate_id()
        now = self._now()
        data_json = json.dumps(data, separators=(",", ":"))
        size_bytes = len(data_json.encode("utf-8"))

        if ttl_days is not None:
            expires_at = now + timedelta(days=ttl_days)
        else:
            expires_at = self._calculate_expiry()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO contexts (id, namespace, data, version, size_bytes,
                                      created_at, updated_at, accessed_at, expires_at)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    context_id,
                    namespace,
                    data_json,
                    size_bytes,
                    now.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )

        return Context(
            id=context_id,
            namespace=namespace,
            data=data,
            version=1,
            size_bytes=size_bytes,
            created_at=now,
            updated_at=now,
            accessed_at=now,
            expires_at=expires_at,
        )

    def get(self, context_id: str, extend_ttl: bool = True) -> Context:
        """
        Get a context by ID.

        Args:
            context_id: Context ID to retrieve
            extend_ttl: If True, extends expiry on access (LRU behavior)

        Returns:
            Context object

        Raises:
            ContextNotFoundError: If context doesn't exist
        """
        self._ensure_initialized()

        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM contexts WHERE id = ?", (context_id,)
        ).fetchone()

        if row is None:
            raise ContextNotFoundError(context_id)

        context = self._row_to_context(row)

        # Check if expired
        if context.expires_at and context.expires_at < self._now():
            # Auto-delete expired context
            self.delete(context_id)
            raise ContextNotFoundError(context_id)

        # Extend TTL on access (LRU behavior)
        if extend_ttl:
            now = self._now()
            new_expiry = self._calculate_expiry()
            with self._transaction() as conn:
                conn.execute(
                    "UPDATE contexts SET accessed_at = ?, expires_at = ? WHERE id = ?",
                    (now.isoformat(), new_expiry.isoformat(), context_id),
                )
            context.accessed_at = now
            context.expires_at = new_expiry

        return context

    def update(
        self,
        context_id: str,
        updates: dict[str, Any],
        version: int,
        merge: bool = True,
    ) -> Context:
        """
        Update a context with optimistic concurrency.

        Args:
            context_id: Context to update
            updates: Data updates to apply
            version: Expected current version (for optimistic locking)
            merge: If True, merges updates with existing data. If False, replaces.

        Returns:
            Updated Context with new version

        Raises:
            ContextNotFoundError: If context doesn't exist
            VersionConflictError: If version doesn't match (concurrent modification)
        """
        self._ensure_initialized()

        with self._transaction() as conn:
            # Get current context with lock
            row = conn.execute(
                "SELECT * FROM contexts WHERE id = ?", (context_id,)
            ).fetchone()

            if row is None:
                raise ContextNotFoundError(context_id)

            current_version = row["version"]
            if current_version != version:
                raise VersionConflictError(context_id, version, current_version)

            # Merge or replace data
            if merge:
                current_data = json.loads(row["data"])
                new_data = {**current_data, **updates}
            else:
                new_data = updates

            now = self._now()
            new_version = current_version + 1
            data_json = json.dumps(new_data, separators=(",", ":"))
            size_bytes = len(data_json.encode("utf-8"))
            new_expiry = self._calculate_expiry()

            conn.execute(
                """
                UPDATE contexts
                SET data = ?, version = ?, size_bytes = ?,
                    updated_at = ?, accessed_at = ?, expires_at = ?
                WHERE id = ?
                """,
                (
                    data_json,
                    new_version,
                    size_bytes,
                    now.isoformat(),
                    now.isoformat(),
                    new_expiry.isoformat(),
                    context_id,
                ),
            )

        return Context(
            id=context_id,
            namespace=row["namespace"],
            data=new_data,
            version=new_version,
            size_bytes=size_bytes,
            created_at=self._parse_timestamp(row["created_at"]),
            updated_at=now,
            accessed_at=now,
            expires_at=new_expiry,
        )

    def delete(self, context_id: str) -> bool:
        """
        Delete a context.

        Args:
            context_id: Context to delete

        Returns:
            True if deleted, False if not found
        """
        self._ensure_initialized()

        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM contexts WHERE id = ?", (context_id,)
            )
            return cursor.rowcount > 0

    def list(
        self,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[ContextSummary]:
        """
        List contexts, optionally filtered by namespace.

        Args:
            namespace: Filter by namespace (exact match). If None, lists all.
            limit: Maximum results to return
            offset: Offset for pagination
            include_expired: If True, includes expired contexts

        Returns:
            List of ContextSummary objects
        """
        self._ensure_initialized()

        conn = self._get_connection()

        conditions = []
        params = []

        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)

        if not include_expired:
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(self._now().isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT id, namespace, version, size_bytes, created_at,
                   updated_at, accessed_at, expires_at
            FROM contexts
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def vacuum(self) -> dict[str, Any]:
        """
        Clean up expired and over-quota contexts.

        Performs:
        1. Delete all expired contexts
        2. If over size quota, delete oldest (by accessed_at) until under quota

        Returns:
            Statistics about cleanup
        """
        self._ensure_initialized()

        stats = {
            "expired_deleted": 0,
            "quota_deleted": 0,
            "total_deleted": 0,
            "current_size_bytes": 0,
            "context_count": 0,
        }

        now = self._now()

        with self._transaction() as conn:
            # 1. Delete expired contexts
            cursor = conn.execute(
                "DELETE FROM contexts WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now.isoformat(),),
            )
            stats["expired_deleted"] = cursor.rowcount
            stats["total_deleted"] = cursor.rowcount

            # 2. Check total size
            row = conn.execute(
                "SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as total_size FROM contexts"
            ).fetchone()
            stats["context_count"] = row["count"]
            stats["current_size_bytes"] = row["total_size"]

            # 3. If over quota, delete oldest by accessed_at until under
            if stats["current_size_bytes"] > self.config.max_size_bytes:
                # Get all contexts ordered by accessed_at (oldest first)
                contexts = conn.execute(
                    "SELECT id, size_bytes FROM contexts ORDER BY accessed_at ASC"
                ).fetchall()

                bytes_to_free = stats["current_size_bytes"] - self.config.max_size_bytes
                bytes_freed = 0
                ids_to_delete = []

                for ctx in contexts:
                    if bytes_freed >= bytes_to_free:
                        break
                    ids_to_delete.append(ctx["id"])
                    bytes_freed += ctx["size_bytes"]

                if ids_to_delete:
                    placeholders = ",".join("?" * len(ids_to_delete))
                    cursor = conn.execute(
                        f"DELETE FROM contexts WHERE id IN ({placeholders})",
                        ids_to_delete,
                    )
                    stats["quota_deleted"] = cursor.rowcount
                    stats["total_deleted"] += cursor.rowcount

            # 4. Final stats
            row = conn.execute(
                "SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as total_size FROM contexts"
            ).fetchone()
            stats["context_count"] = row["count"]
            stats["current_size_bytes"] = row["total_size"]

        # Run SQLite VACUUM to reclaim space
        conn = self._get_connection()
        conn.execute("VACUUM")

        return stats

    def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Statistics dictionary
        """
        self._ensure_initialized()

        conn = self._get_connection()

        row = conn.execute(
            """
            SELECT
                COUNT(*) as context_count,
                COALESCE(SUM(size_bytes), 0) as total_size_bytes,
                COUNT(DISTINCT namespace) as namespace_count,
                MIN(created_at) as oldest_created,
                MAX(updated_at) as newest_updated
            FROM contexts
            WHERE expires_at IS NULL OR expires_at > ?
            """,
            (self._now().isoformat(),),
        ).fetchone()

        expired_count = conn.execute(
            "SELECT COUNT(*) FROM contexts WHERE expires_at IS NOT NULL AND expires_at < ?",
            (self._now().isoformat(),),
        ).fetchone()[0]

        return {
            "context_count": row["context_count"],
            "total_size_bytes": row["total_size_bytes"],
            "namespace_count": row["namespace_count"],
            "oldest_created": row["oldest_created"],
            "newest_updated": row["newest_updated"],
            "expired_count": expired_count,
            "max_size_bytes": self.config.max_size_bytes,
            "max_age_days": self.config.max_age_days,
            "quota_usage_percent": round(
                row["total_size_bytes"] / self.config.max_size_bytes * 100, 2
            ) if self.config.max_size_bytes > 0 else 0,
        }

    def close(self):
        """Close database connection for current thread."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None


# Module-level singleton for global access
_context_manager: ContextManager | None = None
_context_manager_lock = threading.Lock()


def get_context_manager(
    working_directory: str | None = None,
    config: ContextManagerConfig | None = None,
) -> ContextManager:
    """
    Get or create the global ContextManager instance.

    Args:
        working_directory: Override working directory for DB path
        config: Override configuration

    Returns:
        ContextManager instance
    """
    global _context_manager

    with _context_manager_lock:
        if _context_manager is None:
            if config is None:
                config = ContextManagerConfig(working_directory=working_directory)
            _context_manager = ContextManager(config)

        return _context_manager


def reset_context_manager():
    """Reset the global ContextManager (for testing)."""
    global _context_manager

    with _context_manager_lock:
        if _context_manager is not None:
            _context_manager.close()
            _context_manager = None
