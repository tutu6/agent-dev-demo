from __future__ import annotations


class DomainError(Exception):
    """Base error for domain-level failures."""


class ParseError(DomainError):
    """Raised when model output cannot be parsed into required JSON schema."""


class UpstreamServiceError(DomainError):
    """Raised when third-party service invocation fails."""
