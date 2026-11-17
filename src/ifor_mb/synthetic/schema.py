from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSchema:
    timestamp: str
    service: str
    request_id: str
    user: str
    message: str
    severity: str
    client: str
    duration: str


ALIASES = {
    "timestamp": {"timestamp", "time", "datetime", "date_time"},
    "service": {"service", "component", "application"},
    "request_id": {"request_id", "requestid", "request", "trace_id"},
    "user": {"user", "user_id", "userid"},
    "message": {"message", "log_message", "content"},
    "severity": {"severity", "level", "log_level"},
    "client": {"client", "client_id", "source"},
    "duration": {"duration", "latency", "response_time"},
}


def resolve_schema(columns) -> SyntheticSchema:
    normalized = {str(column).strip().lower().replace(" ", "_"): str(column) for column in columns}
    resolved = {}
    for field, aliases in ALIASES.items():
        matches = [normalized[name] for name in aliases if name in normalized]
        if not matches:
            raise ValueError(f"Missing required field {field!r}; accepted aliases: {sorted(aliases)}")
        resolved[field] = matches[0]
    return SyntheticSchema(**resolved)
