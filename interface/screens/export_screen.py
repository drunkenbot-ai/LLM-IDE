from __future__ import annotations

# ExportScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class ExportScreenMixin:
    def create_bundle(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            self.export_log.append("Export Bay is available only in the licensed version.")
            return
        """Create a portable model export bundle."""

        self.export_log.append("Creating model bundle...")
        self.export_progress.setValue(15)
        try:
            output = export_project_bundle(Path(self.export_model_dir.text()), Path(self.export_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Bundle created: {output}")
        self.export_status.setText("Export: bundle created")

    def quantize_model(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            return
        """Create a quantized FP16 checkpoint when selected."""

        mode = self.quant_mode.currentText()
        if not mode.startswith("FP16"):
            self.export_log.append("This GGUF quantization target is planned. FP16 checkpoint quantization is available now.")
            return
        checkpoint = Path(self.export_model_dir.text()) / "final_model.pt"
        output = Path(self.export_dir.text()) / "final_model_fp16.pt"
        self.export_log.append("Creating FP16 checkpoint...")
        self.export_progress.setValue(20)
        try:
            result = quantize_checkpoint(checkpoint, output, mode="fp16")
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Quantized checkpoint created: {result}")
        self.export_status.setText("Export: FP16 checkpoint ready")

    def export_hf_package(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            return
        """Create an HF-style MicroGPT package."""

        self.export_log.append("Creating HF-style MicroGPT package...")
        self.export_progress.setValue(20)
        try:
            result = export_hf_microgpt_package(Path(self.export_model_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"HF package created: {result}")
        self.export_log.append("Note: this package is MicroGPT model_type, not a llama.cpp-supported Llama model.")
        self.export_status.setText("Export: HF package ready")

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
