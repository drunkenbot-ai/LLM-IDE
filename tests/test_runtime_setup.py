from __future__ import annotations

import unittest
from unittest.mock import patch

from packaging.runtime_setup import RuntimeChoice, choose_runtime, get_triton_specifier, install_runtime


class RuntimeSelectionTests(unittest.TestCase):
    def test_new_nvidia_driver_selects_cuda_124(self) -> None:
        self.assertEqual(choose_runtime("Windows", 550).profile, "cu124")

    def test_older_supported_driver_selects_cuda_121(self) -> None:
        self.assertEqual(choose_runtime("Linux", 525).profile, "cu121")

    def test_missing_driver_selects_cpu(self) -> None:
        with patch("packaging.runtime_setup._nvidia_driver_major", return_value=None):
            self.assertEqual(choose_runtime("Windows").profile, "cpu")

    def test_macos_selects_cpu(self) -> None:
        self.assertEqual(choose_runtime("Darwin", 600).profile, "cpu")


class TritonSpecifierTests(unittest.TestCase):
    def test_torch_2_5_maps_to_triton_3_1(self) -> None:
        self.assertEqual(get_triton_specifier("2.5.1"), "triton-windows>=3.1.0,<3.2.0")

    def test_torch_2_6_maps_to_triton_3_2(self) -> None:
        self.assertEqual(get_triton_specifier("2.6.0"), "triton-windows>=3.2.0,<3.3.0")

    def test_torch_2_7_maps_to_triton_3_3(self) -> None:
        self.assertEqual(get_triton_specifier("2.7.0"), "triton-windows>=3.3.0,<3.4.0")

    def test_torch_2_10_maps_to_triton_3_6(self) -> None:
        self.assertEqual(get_triton_specifier("2.10.0"), "triton-windows>=3.6.0,<3.7.0")

    def test_torch_2_4_maps_to_triton_under_3_1(self) -> None:
        self.assertEqual(get_triton_specifier("2.4.1"), "triton-windows<3.1.0")

    def test_fallback_for_unrecognized_version(self) -> None:
        self.assertEqual(get_triton_specifier("custom-build"), "triton-windows")


class RuntimeInstallationTests(unittest.TestCase):
    @patch("packaging.runtime_setup.platform.system", return_value="Windows")
    @patch("packaging.runtime_setup._run_pip")
    def test_windows_cuda_installs_both_torch_and_triton(self, mock_run_pip, _mock_platform) -> None:
        choice = RuntimeChoice(profile="cu124", reason="GPU supported")
        install_runtime("python.exe", choice)

        self.assertEqual(mock_run_pip.call_count, 2)
        torch_call = mock_run_pip.call_args_list[0][0]
        self.assertIn("torch==2.5.1", torch_call[1])
        self.assertIn("https://download.pytorch.org/whl/cu124", torch_call[1])

        triton_call = mock_run_pip.call_args_list[1][0]
        self.assertIn("triton-windows>=3.1.0,<3.2.0", triton_call[1])

    @patch("packaging.runtime_setup.platform.system", return_value="Windows")
    @patch("packaging.runtime_setup._run_pip")
    def test_windows_cpu_skips_triton(self, mock_run_pip, _mock_platform) -> None:
        choice = RuntimeChoice(profile="cpu", reason="No GPU detected")
        install_runtime("python.exe", choice)

        self.assertEqual(mock_run_pip.call_count, 1)
        torch_call = mock_run_pip.call_args_list[0][0]
        self.assertIn("torch==2.5.1", torch_call[1])
        self.assertIn("https://download.pytorch.org/whl/cpu", torch_call[1])

    @patch("packaging.runtime_setup.platform.system", return_value="Linux")
    @patch("packaging.runtime_setup._run_pip")
    def test_linux_skips_triton_windows(self, mock_run_pip, _mock_platform) -> None:
        choice = RuntimeChoice(profile="cu124", reason="GPU supported")
        install_runtime("python3", choice)

        self.assertEqual(mock_run_pip.call_count, 1)
        torch_call = mock_run_pip.call_args_list[0][0]
        self.assertIn("torch==2.5.1", torch_call[1])

    @patch("packaging.runtime_setup.platform.system", return_value="Windows")
    @patch("packaging.runtime_setup._run_pip")
    def test_triton_install_failure_is_non_fatal(self, mock_run_pip, _mock_platform) -> None:
        choice = RuntimeChoice(profile="cu124", reason="GPU supported")
        # First call (torch) succeeds, second call (triton) raises
        mock_run_pip.side_effect = [None, RuntimeError("pip network failure")]

        # install_runtime should not raise
        install_runtime("python.exe", choice)
        self.assertEqual(mock_run_pip.call_count, 2)


if __name__ == "__main__":
    unittest.main()

