"""License activation dialog, shown before the main window when unlicensed."""

from __future__ import annotations

from PySide6.QtCore import Qt, QEventLoop, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from engine.license_client import (
    LicenseCheckResult,
    check_license_at_launch,
    store_license_key,
)


class _LicenseCheckThread(QThread):
    """Runs check_license_at_launch off the GUI thread.

    The whole reason this class exists: network I/O (urllib) must never run
    directly on the Qt main thread, or the entire window -- including the
    splash screen, before any dialog is even shown -- appears frozen for as
    long as the request takes (up to the timeout, longer if DNS hangs).
    """

    finished_with_result = Signal(object)

    def __init__(self, app_version: str, server_url: str) -> None:
        """Create the check thread.

        Args:
            app_version: Running app's version.
            server_url: Base URL of the DrunkenBot cloud service.
        """

        super().__init__()
        self._app_version = app_version
        self._server_url = server_url

    def run(self) -> None:
        """Perform the license check and emit the result."""

        result = check_license_at_launch(self._app_version, self._server_url)
        self.finished_with_result.emit(result)


def run_license_check_responsively(app_version: str, server_url: str) -> LicenseCheckResult:
    """Run a license check without freezing the GUI thread.

    Blocks the *calling function* until a result is available (so callers
    can be written in plain sequential style), but does so via a local
    ``QEventLoop`` rather than a direct call -- Qt keeps processing paint
    and input events for the duration, so the window stays responsive and
    any "Checking..." status text actually renders instead of the whole
    app appearing to hang.

    Args:
        app_version: Running app's version.
        server_url: Base URL of the DrunkenBot cloud service.

    Returns:
        License check result.
    """

    loop = QEventLoop()
    thread = _LicenseCheckThread(app_version, server_url)
    captured: dict[str, LicenseCheckResult] = {}

    def _capture(result: LicenseCheckResult) -> None:
        captured["result"] = result
        loop.quit()

    thread.finished_with_result.connect(_capture)
    thread.start()
    loop.exec()
    thread.wait()
    return captured["result"]


class LicenseActivationDialog(QDialog):
    """Blocking dialog that collects a license key and validates it.

    Loops the user through entry -> validate -> (retry on failure or accept
    on success). The dialog only closes with ``Accepted`` once a license
    key has actually passed :func:`check_license_at_launch`.
    """

    def __init__(self, app_version: str, server_url: str, initial_message: str = "") -> None:
        """Create the activation dialog.

        Args:
            app_version: Running app's version, passed through to validation.
            server_url: Base URL of the DrunkenBot cloud service.
            initial_message: Optional message shown above the input field,
                typically the reason the app isn't already licensed.
        """

        super().__init__()
        self._app_version = app_version
        self._server_url = server_url
        self.result_info: LicenseCheckResult | None = None

        self.setWindowTitle("Activate DrunkenBot LLM-IDE")
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        message = QLabel(
            initial_message or "Enter your license key to activate this copy of the IDE."
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("DBIDE-XXXX-XXXX-XXXX-XXXX")
        layout.addWidget(self._key_input)

        self._status = QTextEdit()
        self._status.setReadOnly(True)
        self._status.setMaximumHeight(80)
        self._status.setVisible(False)
        layout.addWidget(self._status)

        button_row = QHBoxLayout()
        self._activate_button = QPushButton("Activate")
        self._activate_button.clicked.connect(self._on_activate_clicked)
        self._exit_button = QPushButton("Exit")
        self._exit_button.clicked.connect(self.reject)
        button_row.addWidget(self._activate_button)
        button_row.addWidget(self._exit_button)
        layout.addLayout(button_row)

        self._key_input.setFocus(Qt.OtherFocusReason)

    def _on_activate_clicked(self) -> None:
        """Validate the entered license key and close on success."""

        license_key = self._key_input.text().strip()
        if not license_key:
            self._show_status("Please enter a license key.")
            return

        self._activate_button.setEnabled(False)
        self._activate_button.setText("Checking...")
        try:
            store_license_key(license_key)
            result = run_license_check_responsively(self._app_version, self._server_url)
        finally:
            self._activate_button.setEnabled(True)
            self._activate_button.setText("Activate")

        if result.valid:
            self.result_info = result
            self.accept()
            return

        self._show_status(result.reason)

    def _show_status(self, message: str) -> None:
        """Display a status/error message in the dialog.

        Args:
            message: Text to display.
        """

        self._status.setVisible(True)
        self._status.setPlainText(message)
