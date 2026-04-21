from dataclasses import dataclass
from enum import Enum


class AssistantState(str, Enum):
    LOCKED = "Locked"
    UNLOCKED = "Unlocked"
    SLEEP = "Sleep"
    LISTENING = "Listening"
    WAKE_WORD_DETECTED = "Wake Word Detected"


@dataclass(frozen=True)
class BypassResult:
    """Result of a state-based bypass action."""

    changed: bool
    affected_module: str
    state_before: AssistantState
    state_after: AssistantState
    should_start_wakeword_listener: bool
    message: str


@dataclass
class AssistantController:
    state: AssistantState = AssistantState.LOCKED
    failed_attempts: int = 0
    wakeword_enabled: bool = False
    last_detected_wakeword: str | None = None

    def on_verified(self) -> None:
        """Transition to unlocked state and start wake word listener."""
        self.state = AssistantState.UNLOCKED
        self.failed_attempts = 0
        self.activate_wakeword_listener()

    def on_failed_verification(self) -> None:
        """Transition to locked state on verification failure."""
        self.state = AssistantState.LOCKED
        self.failed_attempts += 1

    def bypass_unlock(self) -> None:
        """Unlock via PIN bypass and start wake word listener."""
        self.state = AssistantState.UNLOCKED
        self.failed_attempts = 0
        self.activate_wakeword_listener()

    def apply_bypass(self) -> BypassResult:
        """Apply one generalized bypass action based on current system state.

        Returns:
            BypassResult describing what module was affected and transition details.
        """
        state_before = self.state

        state_handlers = {
            AssistantState.LOCKED: self._bypass_user_verification,
            AssistantState.SLEEP: self._bypass_wakeword_module,
            AssistantState.LISTENING: self._bypass_wakeword_detection,
        }

        handler = state_handlers.get(self.state)
        if handler is None:
            return BypassResult(
                changed=False,
                affected_module="none",
                state_before=state_before,
                state_after=self.state,
                should_start_wakeword_listener=False,
                message=f"Bypass not needed in state: {self.state.value}.",
            )

        return handler(state_before)

    def _bypass_user_verification(self, state_before: AssistantState) -> BypassResult:
        """Bypass for User Verification module when system is locked."""
        self.failed_attempts = 0
        self.activate_wakeword_listener()
        return BypassResult(
            changed=True,
            affected_module="user_verification",
            state_before=state_before,
            state_after=self.state,
            should_start_wakeword_listener=True,
            message="Bypass applied: unlocked user verification and moved to listening.",
        )

    def _bypass_wakeword_module(self, state_before: AssistantState) -> BypassResult:
        """Bypass for Wake Word module when system is sleeping."""
        self.activate_wakeword_listener()
        return BypassResult(
            changed=True,
            affected_module="wake_word_detection",
            state_before=state_before,
            state_after=self.state,
            should_start_wakeword_listener=True,
            message="Bypass applied: woke from sleep and resumed listening.",
        )

    def _bypass_wakeword_detection(self, state_before: AssistantState) -> BypassResult:
        """Bypass wake word detection while already listening."""
        self.on_wakeword_detected("Bypass")
        return BypassResult(
            changed=True,
            affected_module="wake_word_detection",
            state_before=state_before,
            state_after=self.state,
            should_start_wakeword_listener=False,
            message="Bypass applied: wake word accepted while listening.",
        )

    def activate_wakeword_listener(self) -> None:
        """Start listening for wake word."""
        self.state = AssistantState.LISTENING
        self.wakeword_enabled = True

    def on_wakeword_detected(self, wakeword: str = "Hey Atlas") -> None:
        """Handle wake word detection."""
        self.state = AssistantState.WAKE_WORD_DETECTED
        self.last_detected_wakeword = wakeword

    def on_asr_confirmed(self) -> None:
        """Return to listening after ASR confirmation."""
        self.activate_wakeword_listener()

    def on_listening_timeout(self) -> None:
        """Transition to sleep after listening timeout."""
        self.state = AssistantState.SLEEP
        self.wakeword_enabled = False

    def return_to_locked(self) -> None:
        """Return to locked state."""
        self.state = AssistantState.LOCKED
        self.failed_attempts = 0
        self.wakeword_enabled = False
        self.last_detected_wakeword = None
