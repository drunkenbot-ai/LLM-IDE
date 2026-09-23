from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox

from engine.config import ModelConfig, TrainingConfig
from engine.model_gpt import MicroGPT
from interface import app as interface_app


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class DummyAppWindow(interface_app.TrainingScreenMixin, interface_app.TrainingConfigMixin, interface_app.FineTuningScreenMixin):
    def __init__(self):
        self.train_context_length = QSpinBox()
        self.train_context_length.setRange(16, 1000000)
        self.train_context_length.setValue(1024)
        self.n_embd = QSpinBox()
        self.n_embd.setRange(16, 100000)
        self.n_embd.setValue(512)
        self.n_head = QSpinBox()
        self.n_head.setRange(1, 1024)
        self.n_head.setValue(8)
        self.n_layer = QSpinBox()
        self.n_layer.setRange(1, 1024)
        self.n_layer.setValue(4)
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 1.0)
        self.dropout.setValue(0.0)
        self.use_bias = QCheckBox()
        self.use_bias.setChecked(False)

        self.intermediate_size = QSpinBox()
        self.intermediate_size.setRange(0, 1000000)
        self.intermediate_size.setValue(14336)
        self.tie_embeddings = QCheckBox()
        self.tie_embeddings.setChecked(True)

        self.attention_type = QComboBox()
        self.attention_type.addItems(["Multi-Head (MHA)", "Grouped-Query (GQA)", "Multi-Query (MQA)"])

        self.kv_head_count = QSpinBox()
        self.kv_head_count.setRange(1, 1024)
        self.kv_head_count.setValue(2)

        self.attention_backend = QComboBox()
        self.attention_backend.addItems(["sdpa", "manual"])

        self.attention_window = QSpinBox()
        self.attention_window.setRange(0, 1000000)
        self.attention_window.setValue(0)

        self.rope_theta = QDoubleSpinBox()
        self.rope_theta.setRange(0.0, 10000000.0)
        self.rope_theta.setValue(500000.0)

        self.norm_type = QComboBox()
        self.norm_type.addItems(["RMSNorm", "LayerNorm"])

        self.activation_fn = QComboBox()
        self.activation_fn.addItems(["SwiGLU", "GELU", "SiLU"])

        self.architecture_style = QComboBox()
        self.architecture_style.addItems(["Llama-like", "Classic GPT"])

        self.target_self_attn = QPushButton()
        self.target_self_attn.setCheckable(True)
        self.target_self_attn.setChecked(True)

        self.target_mlp = QPushButton()
        self.target_mlp.setCheckable(True)
        self.target_mlp.setChecked(True)

        self.lora_targets = QComboBox()
        self.lora_targets.addItems(["Attention projections", "MLP projections", "Attention + MLP"])

        self.peft_method = QComboBox()
        self.peft_method.addItems(["Full fine-tune", "LoRA adapters"])

        self.training_launch_target = QComboBox()
        self.training_launch_target.addItems(["Local Process", "Cluster Fleet"])

        self.training_mode = QComboBox()
        self.training_mode.addItems(["Pretrain from scratch", "Fine-tune checkpoint"])

        self.resume_checkpoint = QLineEdit()
        self.fine_tune_checkpoint = QLineEdit()
        self.model_dir = QLineEdit("runs/model")

    def _set_combo_text(self, combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)


def test_attention_type_value(qapp):
    win = DummyAppWindow()
    win.attention_type.setCurrentText("Multi-Head (MHA)")
    assert win._attention_type_value() == "mha"

    win.attention_type.setCurrentText("Grouped-Query (GQA)")
    assert win._attention_type_value() == "gqa"

    win.attention_type.setCurrentText("Multi-Query (MQA)")
    assert win._attention_type_value() == "mqa"


def test_architecture_style_config(qapp):
    win = DummyAppWindow()
    win.norm_type.setCurrentText("RMSNorm")
    win.activation_fn.setCurrentText("SwiGLU")
    win.rope_theta.setValue(500000.0)

    cfg = win._architecture_style_config()
    assert cfg["norm_type"] == "rmsnorm"
    assert cfg["mlp_type"] == "swiglu"
    assert cfg["position_encoding"] == "rope"
    assert cfg["rope_theta"] == 500000.0

    win.norm_type.setCurrentText("LayerNorm")
    win.activation_fn.setCurrentText("GELU")
    win.rope_theta.setValue(0.0)
    cfg2 = win._architecture_style_config()
    assert cfg2["norm_type"] == "layernorm"
    assert cfg2["mlp_type"] == "gelu"
    assert cfg2["position_encoding"] == "learned"


def test_current_model_config_intermediate_and_tie(qapp):
    win = DummyAppWindow()
    win.attention_type.setCurrentText("Grouped-Query (GQA)")
    win.kv_head_count.setValue(2)
    win.intermediate_size.setValue(14336)
    win.tie_embeddings.setChecked(False)
    win.norm_type.setCurrentText("RMSNorm")
    win.activation_fn.setCurrentText("SwiGLU")

    m_cfg = win._current_model_config(vocab_size=32000)
    assert m_cfg.attention_type == "gqa"
    assert m_cfg.kv_head_count == 2
    assert m_cfg.intermediate_size == 14336
    assert m_cfg.tie_word_embeddings is False
    assert m_cfg.norm_type == "rmsnorm"
    assert m_cfg.mlp_type == "swiglu"


def test_micro_gpt_untied_embeddings(qapp):
    config = ModelConfig(
        vocab_size=1000,
        context_length=64,
        embedding_size=64,
        head_count=4,
        layer_count=2,
        dropout=0.0,
        bias=False,
        norm_type="rmsnorm",
        position_encoding="rope",
        mlp_type="swiglu",
        intermediate_size=128,
        attention_type="gqa",
        kv_head_count=2,
        tie_word_embeddings=False,
    )
    model = MicroGPT(config)
    assert model.token_embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()

    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 1000)
    loss = logits.sum()
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_micro_gpt_tied_embeddings(qapp):
    config = ModelConfig(
        vocab_size=1000,
        context_length=64,
        embedding_size=64,
        head_count=4,
        layer_count=2,
        dropout=0.0,
        bias=False,
        norm_type="rmsnorm",
        position_encoding="rope",
        mlp_type="swiglu",
        intermediate_size=128,
        attention_type="gqa",
        kv_head_count=2,
        tie_word_embeddings=True,
    )
    model = MicroGPT(config)
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_lora_target_value_pills(qapp):
    win = DummyAppWindow()
    win.target_self_attn.setChecked(True)
    win.target_mlp.setChecked(True)
    assert win._lora_target_value() == "attention,mlp"

    win.target_self_attn.setChecked(False)
    win.target_mlp.setChecked(True)
    assert win._lora_target_value() == "mlp"

    win.target_self_attn.setChecked(True)
    win.target_mlp.setChecked(False)
    assert win._lora_target_value() == "attention"
