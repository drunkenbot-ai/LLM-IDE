from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart5:
    def _tip(self, widget: QWidget, text: str) -> None:
        """Attach tooltip and status tip text.

        Args:
            widget: Widget receiving the tip.
            text: Tooltip text.
        """

        widget.setToolTip(text)
        widget.setStatusTip(text)

    def _render_chat_markdown(self, markdown_text: str) -> None:
        """Render chat Markdown with highlighted fenced code blocks when possible.

        Args:
            markdown_text: Markdown transcript to render.
        """

        if not hasattr(self, "current_assistant_message") or self.current_assistant_message is None:
            return
        self.current_assistant_message.set_content(markdown_text)

    def _add_chat_message(
        self,
        role: str,
        content: str,
        metrics: str = "",
        resend_prompt: Optional[str] = None,
    ) -> QTextBrowser:
        """Add one chat bubble.

        Args:
            role: Message role, either ``user`` or ``assistant``.
            content: Markdown message content.
            metrics: Optional metric text shown under assistant replies.
            resend_prompt: Prompt to resend from the bubble.

        Returns:
            Text browser used by the bubble.
        """

        should_follow = self._is_chat_near_bottom()
        max_width = max(320, int(self.chat_scroll.viewport().width() * 0.78)) if hasattr(self, "chat_scroll") else 900
        message = ChatMessageWidget(
            role,
            content,
            markdown_to_html,
            self._resend_chat_message,
            metrics=metrics,
            resend_prompt=resend_prompt,
            max_width=max_width,
        )
        self.chat_messages.insertWidget(max(self.chat_messages.count() - 1, 0), message)
        if should_follow:
            message.scroll_later(lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))
        if role == "assistant":
            self.current_assistant_message = message
            self.current_assistant_browser = message.browser
            self.current_assistant_meta = message.meta_label
        return message.browser

    def _is_chat_near_bottom(self) -> bool:
        """Return whether the chat scroll is close enough to follow streaming.

        Returns:
            True when the view should auto-scroll.
        """

        if not hasattr(self, "chat_scroll"):
            return True
        bar = self.chat_scroll.verticalScrollBar()
        return bar.maximum() - bar.value() < 48

    def _clear_chat_messages(self) -> None:
        """Remove all message bubbles."""

        while self.chat_messages.count() > 1:
            item = self.chat_messages.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.current_assistant_message = None
        self.current_assistant_browser = None
        self.current_assistant_meta = None

    def _resend_chat_message(self, prompt: str) -> None:
        """Resend text from a message bubble.

        Args:
            prompt: Prompt text to send.
        """

        self.chat_input.setPlainText(prompt)
        self.send_chat_message()

    def _set_chat_stats(self, elapsed_seconds: float, token_count: int, tokens_per_second: float) -> None:
        """Update live chat generation metrics.

        Args:
            elapsed_seconds: Elapsed generation time.
            token_count: Generated token count.
            tokens_per_second: Approximate token speed.
        """

        text = f"Time: {elapsed_seconds:.2f}s  |  Tokens: {token_count:,}  |  Speed: {tokens_per_second:.2f} tok/s"
        self.chat_stats.setText(text)
        if self.current_assistant_meta is not None:
            self.current_assistant_meta.setText(text)
            self.current_assistant_meta.setVisible(True)

    def _chat_backend_value(self) -> str:
        """Return the selected chat model backend.

        Returns:
            Stable chat backend identifier.
        """

        if not hasattr(self, "chat_model_backend"):
            return "gguf"
        return "microgpt" if self.chat_model_backend.currentText() == "MicroGPT checkpoint" else "gguf"

    def _update_chat_backend_controls(self) -> None:
        """Show controls relevant to the selected chat backend."""

        if not hasattr(self, "chat_model_backend"):
            return
        native = self._chat_backend_value() == "microgpt"
        self.gguf_path_row.setVisible(not native)
        self.microgpt_path_row.setVisible(native)
        self.llama_gpu_layers.setEnabled(not native)
        self.llama_threads.setEnabled(not native)
        self.llama_context.setEnabled(not native)
        if native:
            self._tip(self.load_llm_button, "Load the native MicroGPT checkpoint into memory once for repeated chat messages.")
        else:
            self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")

    def _app_icon(self) -> QIcon:
        """Create the application icon.

        Returns:
            Application icon.
        """

        return self._static_app_icon()

    @staticmethod
    def _app_logo_path() -> Path:
        """Return the bundled logo path.

        Returns:
            Logo path.
        """

        candidates = [
            Path(__file__).resolve().parent / "drunken_bot_logo_small.png",
        ]
        if hasattr(sys, "_MEIPASS"):
            candidates.insert(0, Path(sys._MEIPASS) / "drunken_bot_logo_small.png")
        app_root = os.environ.get("DRUNKENBOT_APP_ROOT")
        if app_root:
            candidates.insert(0, Path(app_root) / "interface" / "drunken_bot_logo_small.png")
        return next((path for path in candidates if path.exists()), candidates[0])

    @staticmethod
    def _app_logo_pixmap(size: int = 64) -> QPixmap:
        """Load the bundled logo as a pixmap.

        Args:
            size: Maximum square size.

        Returns:
            Logo pixmap, or null pixmap when the file is missing.
        """

        pixmap = QPixmap(str(MainWindowPart5._app_logo_path()))
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    @staticmethod
    def _static_app_icon() -> QIcon:
        """Create the static app icon.

        Returns:
            Application icon.
        """

        logo_path = MainWindowPart5._app_logo_path()
        if logo_path.exists():
            icon = QIcon(str(logo_path))
            if not icon.isNull():
                return icon
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor("#1f1f1f")))
            painter.setPen(QPen(QColor("#f5b041"), 3))
            painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
            bolt = QPolygon([
                QPoint(36, 8),
                QPoint(17, 35),
                QPoint(31, 35),
                QPoint(25, 56),
                QPoint(48, 25),
                QPoint(33, 25),
            ])
            painter.setPen(QPen(QColor("#ffd27a"), 2))
            painter.setBrush(QBrush(QColor("#f5b041")))
            painter.drawPolygon(bolt)
        finally:
            painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _windows_icon_path() -> Path:
        """Return the Windows icon file path.

        Returns:
            Path to the generated ``.ico`` file.
        """

        return Path(__file__).with_name("drunkenbot_llm_ide.ico")

    @staticmethod
    def _ensure_windows_icon_file() -> Optional[Path]:
        """Ensure the generated Windows ``.ico`` file exists.

        Returns:
            Icon path on Windows, otherwise ``None``.
        """

        if sys.platform != "win32":
            return None
        icon_path = MainWindowPart5._windows_icon_path()
        if icon_path.exists():
            return icon_path
        icon = MainWindowPart5._static_app_icon()
        pixmap = icon.pixmap(256, 256)
        if pixmap.isNull() or not pixmap.save(str(icon_path), "ICO"):
            return None
        return icon_path

    def apply_windows_taskbar_icon(self) -> None:
        """Apply the app icon to the native Windows window handle."""

        if sys.platform != "win32":
            return
        icon_path = self._ensure_windows_icon_file()
        if icon_path is None:
            return

        hwnd = int(self.winId())
        if not hwnd:
            return

        wm_seticon = 0x0080
        icon_small = 0
        icon_big = 1
        image_icon = 1
        lr_loadfromfile = 0x0010

        user32 = ctypes.windll.user32
        hicon_big = user32.LoadImageW(None, str(icon_path), image_icon, 256, 256, lr_loadfromfile)
        hicon_small = user32.LoadImageW(None, str(icon_path), image_icon, 32, 32, lr_loadfromfile)
        if hicon_big:
            user32.SendMessageW(hwnd, wm_seticon, icon_big, hicon_big)
            self._windows_icon_handles.append(hicon_big)
        if hicon_small:
            user32.SendMessageW(hwnd, wm_seticon, icon_small, hicon_small)
            self._windows_icon_handles.append(hicon_small)

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str = "Checkpoints (*.pt)") -> None:
        """Open a file or folder picker for a path field.

        Args:
            field: Path input to update.
            directory: Whether to select a folder instead of a file.
            file_filter: File dialog filter used for files.
        """

        start_dir = self._browse_start_dir(field, directory)
        if directory:
            value = QFileDialog.getExistingDirectory(self, "Choose folder", start_dir)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Choose file", start_dir, file_filter)
        if value:
            field.setText(value)

    def _browse_multiple_files(self, field: QLineEdit, file_filter: str) -> None:
        """Open a multi-file picker and write selected paths to a field.

        Args:
            field: Path field to update.
            file_filter: File dialog filter.
        """

        values, _ = QFileDialog.getOpenFileNames(self, "Choose files", self._browse_start_dir(field, False), file_filter)
        if values:
            field.setText("; ".join(values))

    def _browse_start_dir(self, field: QLineEdit, directory: bool) -> str:
        """Return the best initial folder for a browse dialog.

        Args:
            field: Path field being browsed.
            directory: Whether the dialog selects a folder.

        Returns:
            Existing field path, active project folder, or current folder.
        """

        text = field.text().strip()
        if text:
            path = Path(text)
            if path.exists():
                if path.is_dir():
                    return str(path)
                return str(path.parent)
            parent = path if directory else path.parent
            if parent.exists():
                return str(parent)
        if self.current_project_file is not None:
            return str(self.current_project_file.parent)
        return str(Path.cwd())

    def save_project(self) -> None:
        """Save the current project settings into a named project folder."""

        project_name = self.search_box.text().strip() or "DrunkenBotProject"
        safe_name = self._safe_project_name(project_name)
        if self.current_project_file is None:
            base_dir = QFileDialog.getExistingDirectory(self, "Choose parent folder for project", self._project_dialog_start_dir())
            if not base_dir:
                return
            project_dir = Path(base_dir) / safe_name
            project_file = project_dir / "project.json"
        else:
            project_file = self.current_project_file
            project_dir = project_file.parent
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        if self.current_project_file is None:
            self._apply_project_workspace_paths(project_dir)
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        self.current_project_file = project_file
        _register_recent_project(project_file)
        self._apply_project_runtime_environment(project_dir)
        self._refresh_notification_manager(project_dir)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        self.project_state.setText("Project saved")
        LOGGER.info("Project saved: %s", project_file)
        if self.current_project_file == project_file:
            self.dataset_log.append(f"Project saved: {project_file}")
            self.dataset_log.append(f"Project workspace: {project_dir}")
            self.dataset_log.append(f"Notifier config: {project_dir / 'notifier_config.json'}")

    def new_project(self) -> None:
        """Start a fresh project and clear the active project file binding."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please stop or wait for the current task before creating a new project.")
            return
        if self.current_project_file is not None or self.search_box.text().strip():
            choice = QMessageBox.question(
                self,
                "New project",
                "Start a new project? Unsaved changes in the current project will not be saved automatically.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        base_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose folder where the new project will be created",
            self._project_dialog_start_dir(),
        )
        if not base_dir:
            self.project_state.setText("Project creation cancelled")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        project_name = self.search_box.text().strip() or "DrunkenBotProject"
        try:
            self._create_project_at(project_name, Path(base_dir))
        except Exception as exc:
            QMessageBox.warning(self, "New project failed", f"Could not create project:\n{exc}")

    def open_project(self) -> None:
        """Open a saved project file and restore UI settings."""

        project_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Micro LLM project",
            self._project_dialog_start_dir(),
            "Micro LLM project (project.json *.json);;All files (*)",
        )
        if not project_file:
            return
        try:
            self._open_project_file(Path(project_file))
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open project:\n{exc}")
            return
