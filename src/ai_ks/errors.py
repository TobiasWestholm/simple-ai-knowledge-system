from __future__ import annotations


class DependencyUnavailableError(RuntimeError):
    def __init__(self, service: str, message: str) -> None:
        super().__init__(message)
        self.service = service
