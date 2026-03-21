"""
Unit tests for app.utils.retry
"""

import time
import pytest
from unittest.mock import MagicMock

from app.utils.retry import retry_with_backoff, RetryableAPIClient


# ---------------------------------------------------------------------------
# retry_with_backoff decorator
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:
    def test_no_retry_on_success(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = always_succeeds()
        assert result == "ok"
        assert call_count == 1

    def test_retries_and_eventually_succeeds(self):
        attempts = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def fails_twice_then_succeeds():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return "success"

        result = fails_twice_then_succeeds()
        assert result == "success"
        assert len(attempts) == 3

    def test_raises_after_max_retries(self):
        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def always_fails():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            always_fails()

    def test_total_calls_equal_max_retries_plus_one(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 4  # 1 initial + 3 retries

    def test_only_retries_specified_exceptions(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raises_type_error()

        # Should NOT retry for TypeError (only ValueError is specified)
        assert call_count == 1

    def test_on_retry_callback_is_called(self):
        callback = MagicMock()

        @retry_with_backoff(max_retries=2, initial_delay=0.01, on_retry=callback)
        def always_fails():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fails()

        assert callback.call_count == 2

    def test_preserves_function_name(self):
        @retry_with_backoff(max_retries=1, initial_delay=0.01)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_returns_value_on_first_retry(self):
        attempts = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def fails_once():
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("first attempt")
            return 42

        assert fails_once() == 42


# ---------------------------------------------------------------------------
# RetryableAPIClient
# ---------------------------------------------------------------------------

class TestRetryableAPIClient:
    def test_call_with_retry_success(self):
        client = RetryableAPIClient(max_retries=2, initial_delay=0.01)
        result = client.call_with_retry(lambda x: x * 2, 5)
        assert result == 10

    def test_call_with_retry_raises_after_max(self):
        client = RetryableAPIClient(max_retries=2, initial_delay=0.01)
        with pytest.raises(RuntimeError):
            client.call_with_retry(
                lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                exceptions=(RuntimeError,)
            )

    def test_call_batch_with_retry_all_success(self):
        client = RetryableAPIClient(max_retries=1, initial_delay=0.01)
        items = [1, 2, 3]
        results, failures = client.call_batch_with_retry(items, lambda x: x ** 2)
        assert results == [1, 4, 9]
        assert failures == []

    def test_call_batch_with_retry_handles_failures(self):
        client = RetryableAPIClient(max_retries=0, initial_delay=0.01)

        def bad_func(x):
            if x == 2:
                raise ValueError("skip 2")
            return x

        results, failures = client.call_batch_with_retry(
            [1, 2, 3],
            bad_func,
            exceptions=(ValueError,),
            continue_on_failure=True
        )
        assert 1 in results
        assert 3 in results
        assert len(failures) == 1
        assert failures[0]["index"] == 1
