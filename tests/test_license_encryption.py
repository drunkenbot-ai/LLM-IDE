"""Tests for machine-bound encrypted local licensing and startup checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import InvalidToken

from engine.license_client import (
    LicenseCheckResult,
    activate_license,
    check_license_at_launch,
    check_local_license,
    clear_encrypted_license,
    decrypt_license_metadata,
    encrypt_license_metadata,
    load_encrypted_license,
    load_stored_license_key,
    save_encrypted_license,
    store_license_key,
    _derive_machine_key,
    _get_or_create_machine_id,
)


@pytest.fixture
def temp_license_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate license directory in a temporary folder."""
    license_dir = tmp_path / ".drunkenbot_ide" / "license"
    license_dir.mkdir(parents=True, exist_ok=True)

    key_file = license_dir / "license_key.txt"
    vault_file = license_dir / "license_vault.enc"
    grace_file = license_dir / "grace_receipt.json"
    machine_id_file = license_dir / "machine_id.txt"

    monkeypatch.setattr("engine.license_client.LICENSE_DIR", license_dir)
    monkeypatch.setattr("engine.license_client.LICENSE_KEY_FILE", key_file)
    monkeypatch.setattr("engine.license_client.LICENSE_VAULT_FILE", vault_file)
    monkeypatch.setattr("engine.license_client.GRACE_CACHE_FILE", grace_file)
    monkeypatch.setattr("engine.license_client.MACHINE_ID_FILE", machine_id_file)

    return {
        "dir": license_dir,
        "key_file": key_file,
        "vault_file": vault_file,
        "grace_file": grace_file,
        "machine_id_file": machine_id_file,
    }


def test_encryption_roundtrip(temp_license_env):
    """Verify that metadata can be encrypted and decrypted accurately."""
    machine_id = _get_or_create_machine_id()
    payload = {
        "license_key": "DBIDE-TEST-1234-5678-ABCD",
        "machine_id": machine_id,
        "valid": True,
        "version_ceiling": "999.0.0",
        "custom_info": {"tier": "pro"},
    }

    encrypted = encrypt_license_metadata(payload, machine_id)
    assert isinstance(encrypted, bytes)
    assert len(encrypted) > 0
    assert not b"DBIDE-TEST-1234-5678-ABCD" in encrypted  # Key is encrypted, not plaintext

    decrypted = decrypt_license_metadata(encrypted, machine_id)
    assert decrypted == payload


def test_encryption_tamper_and_machine_binding(temp_license_env):
    """Verify decryption fails if machine ID is different or ciphertext is corrupted."""
    mid_1 = "machine_one"
    mid_2 = "machine_two"
    payload = {"license_key": "DBIDE-SECRET-KEY", "valid": True}

    encrypted = encrypt_license_metadata(payload, mid_1)

    # Attempt decrypt with different machine ID
    with pytest.raises((InvalidToken, Exception)):
        decrypt_license_metadata(encrypted, mid_2)

    # Attempt decrypt with corrupted bytes
    corrupted = bytearray(encrypted)
    corrupted[-5] ^= 0xFF
    with pytest.raises((InvalidToken, Exception)):
        decrypt_license_metadata(bytes(corrupted), mid_1)


def test_save_and_load_encrypted_license(temp_license_env):
    """Test saving and loading encrypted license vault."""
    machine_id = _get_or_create_machine_id()
    metadata = {
        "license_key": "DBIDE-VALID-KEY-0001",
        "machine_id": machine_id,
        "valid": True,
        "version_ceiling": "2.0.0",
    }

    assert load_encrypted_license() is None
    save_encrypted_license(metadata)
    assert temp_license_env["vault_file"].exists()

    loaded = load_encrypted_license()
    assert loaded == metadata
    assert load_stored_license_key() == "DBIDE-VALID-KEY-0001"

    clear_encrypted_license()
    assert load_encrypted_license() is None


def test_check_local_license_validity(temp_license_env):
    """Test check_local_license under various states."""
    app_version = "1.4.0"
    machine_id = _get_or_create_machine_id()

    # 1. No local license
    result = check_local_license(app_version)
    assert not result.valid

    # 2. Valid local license
    save_encrypted_license({
        "license_key": "DBIDE-TEST-KEY",
        "machine_id": machine_id,
        "valid": True,
        "version_ceiling": "2.0.0",
    })
    result = check_local_license(app_version)
    assert result.valid
    assert "local encrypted license" in result.reason

    # 3. Machine mismatch
    save_encrypted_license({
        "license_key": "DBIDE-TEST-KEY",
        "machine_id": "other_machine_id",
        "valid": True,
        "version_ceiling": "2.0.0",
    })
    result = check_local_license(app_version)
    assert not result.valid
    assert "different machine" in result.reason

    # 4. Version ceiling exceeded
    save_encrypted_license({
        "license_key": "DBIDE-TEST-KEY",
        "machine_id": machine_id,
        "valid": True,
        "version_ceiling": "1.2.0",
    })
    result = check_local_license(app_version)
    assert not result.valid
    assert "covers up to version" in result.reason

    # 5. Grace period expired
    expired_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    save_encrypted_license({
        "license_key": "DBIDE-TEST-KEY",
        "machine_id": machine_id,
        "valid": True,
        "version_ceiling": "2.0.0",
        "grace_period_until": expired_time,
    })
    result = check_local_license(app_version)
    assert not result.valid
    assert "validity period has expired" in result.reason


def test_check_license_at_launch_local_precedence(temp_license_env):
    """Test that check_license_at_launch prioritizes valid local license without network."""
    app_version = "1.4.0"
    machine_id = _get_or_create_machine_id()

    save_encrypted_license({
        "license_key": "DBIDE-TEST-KEY",
        "machine_id": machine_id,
        "valid": True,
        "version_ceiling": "2.0.0",
    })

    with patch("engine.license_client._validate_online") as mock_online:
        result = check_license_at_launch(app_version, "https://drunkenbot.store")
        assert result.valid
        assert "local encrypted license" in result.reason
        # Network validation MUST NOT be called when local license is valid
        mock_online.assert_not_called()


def test_check_license_at_launch_falls_back_to_online(temp_license_env):
    """Test that check_license_at_launch falls back to online validation when local vault missing."""
    app_version = "1.4.0"
    store_license_key("DBIDE-ONLINE-KEY")

    mock_response = {
        "valid": True,
        "version_ceiling": "3.0.0",
        "grace_period_until": None,
        "receipt": None,
        "signature": None,
    }

    with patch("engine.license_client._validate_online", return_value=mock_response) as mock_online:
        result = check_license_at_launch(app_version, "https://drunkenbot.store")
        assert result.valid
        assert result.reason == "Validated online."
        mock_online.assert_called_once()

        # Should have saved the encrypted license locally
        loaded = load_encrypted_license()
        assert loaded is not None
        assert loaded["license_key"] == "DBIDE-ONLINE-KEY"
        assert loaded["valid"] is True


def test_activate_license_flow(temp_license_env):
    """Test activate_license saves encrypted vault upon successful validation."""
    app_version = "1.4.0"
    mock_response = {
        "valid": True,
        "version_ceiling": "5.0.0",
        "grace_period_until": None,
        "receipt": "receipt_data",
        "signature": "sig_data",
    }

    with patch("engine.license_client._validate_online", return_value=mock_response):
        result = activate_license("DBIDE-NEW-KEY", app_version, "https://drunkenbot.store")
        assert result.valid
        assert result.reason == "License activated successfully."

        # Verify encrypted license is stored
        loaded = load_encrypted_license()
        assert loaded is not None
        assert loaded["license_key"] == "DBIDE-NEW-KEY"
        assert loaded["valid"] is True
        assert loaded["version_ceiling"] == "5.0.0"

    # Test rejected activation
    reject_response = {"valid": False, "reason": "Revoked or invalid."}
    with patch("engine.license_client._validate_online", return_value=reject_response):
        result = activate_license("DBIDE-BAD-KEY", app_version, "https://drunkenbot.store")
        assert not result.valid
        assert result.reason == "Revoked or invalid."
        assert load_encrypted_license() is None
