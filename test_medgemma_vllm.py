import unittest

from medgemma_vllm import _wait_for_server


class ExitedProcess:
    returncode = 23

    def poll(self):
        return self.returncode

    def terminate(self):
        raise AssertionError("an exited process must not be terminated again")


class WaitForServerTest(unittest.TestCase):
    def test_reports_vllm_exit_instead_of_waiting_for_startup_timeout(self):
        with self.assertRaisesRegex(RuntimeError, "vLLM exited during startup with status 23"):
            _wait_for_server(ExitedProcess(), port=1, timeout_s=60, poll_interval_s=0)


if __name__ == "__main__":
    unittest.main()
