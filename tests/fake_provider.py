"""A provider double that records exactly what DelegateDoctor tried to send.

The privacy tests in this suite are only meaningful if they can inspect the
real outbound payload, so every test that involves AI goes through this. It
records requests verbatim - before any assertion - so a test can make both
positive claims ("the source really was sent") and negative ones ("the API key
was not").
"""

from delegate_doctor.agent.client import AIError, AIRequest, AIResponse, Provider
from delegate_doctor.agent.provider_response import (SUCCESS,
                                                     ProviderCompletionResult,
                                                     error_result)


class FakeProvider(Provider):
    """Replays canned replies and remembers every request it was given."""

    name = "fake"

    def __init__(self, *replies, error=None):
        self.replies = list(replies)
        self.error = error
        self.requests = []

    def complete(self, request: AIRequest) -> ProviderCompletionResult:
        """The authoritative method, as the real provider implements it.

        `complete_structured` is inherited, so a double exercises the same
        success/failure translation production does.
        """
        self.requests.append(request)
        if self.error is not None:
            return error_result(str(self.error), exception=self.error)
        if not self.replies:
            raise AssertionError(
                f"the provider was called {len(self.requests)} times but only "
                f"{len(self.requests) - 1} replies were scripted"
            )
        return ProviderCompletionResult(SUCCESS, text=self.replies.pop(0),
                                        diagnostics={"model": "fake-model"})

    # --- what tests assert against -----------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def sent_text(self) -> str:
        """Every character of every request, concatenated."""
        return "\n".join(request.payload_text() for request in self.requests)

    def assert_never_sent(self, *fragments) -> None:
        payload = self.sent_text
        for fragment in fragments:
            assert fragment not in payload, (
                f"DelegateDoctor sent something it must never send: {fragment!r}")

    def assert_sent(self, *fragments) -> None:
        payload = self.sent_text
        for fragment in fragments:
            assert fragment in payload, (
                f"expected {fragment!r} in the outbound request")


class RefusingProvider(Provider):
    """Fails the test if it is called at all."""

    name = "refusing"

    def __init__(self, reason="the provider must not be called"):
        self.reason = reason

    def complete(self, request):
        raise AssertionError(self.reason)
