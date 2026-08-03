from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart18:
    def export_llama_adapter(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            return
        """Create a directly loadable Llama-family package when compatible."""

        self.export_log.append("Creating Llama-compatible adapter package...")
        self.export_progress.setValue(20)
        try:
            result = export_llama_adapter_package(Path(self.export_model_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Llama adapter package created: {result}")
        self.export_status.setText("Export: Llama adapter ready")

    def convert_hf_to_gguf(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            return
        """Convert an HF-compatible model folder to GGUF through llama.cpp."""

        model_dir_text = self.export_model_dir.text().strip()
        llama_dir_text = self.llama_cpp_dir.text().strip()
        output_text = self.gguf_output_path.text().strip()
        if not model_dir_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose the model core folder first.")
            return
        if not (Path(model_dir_text) / "hf_model").exists():
            QMessageBox.warning(
                self,
                "GGUF blocked",
                "GGUF conversion needs an HF model package first. Use Export HF Package, then convert a llama.cpp-supported model.",
            )
            return
        if not llama_dir_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose your local llama.cpp folder containing convert_hf_to_gguf.py.")
            return
        if not output_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose a GGUF output file path.")
            return
        self.export_log.append("Starting llama.cpp GGUF conversion...")
        self.export_progress.setValue(0)
        self._run_task(
            export_gguf_with_llama_cpp,
            (
                Path(model_dir_text),
                Path(llama_dir_text),
                Path(output_text),
                self.gguf_outtype.currentText(),
            ),
            self._gguf_conversion_finished,
            self.export_log,
            self.export_progress,
            button=self.gguf_convert_button,
            busy_text="Converting GGUF",
        )

    @Slot(object)
    def _gguf_conversion_finished(self, result: Any) -> None:
        """Update UI after GGUF conversion finishes.

        Args:
            result: GGUF output path.
        """

        self.export_progress.setValue(100)
        self.export_log.append(f"GGUF created: {result}")
        self.gguf_path.setText(str(result))
        self.export_status.setText("Export: GGUF ready")
        self._clear_button_busy("Convert HF to GGUF")

    def _apply_preset(self, preset: str) -> None:
        """Apply architecture values for a preset.

        Args:
            preset: Selected preset name.
        """

        if preset == "Tiny":
            self.n_embd.setValue(128)
            self.n_head.setValue(4)
            self.n_layer.setValue(4)
        elif preset == "Small":
            self.n_embd.setValue(512)
            self.n_head.setValue(8)
            self.n_layer.setValue(8)


