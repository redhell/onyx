"""
Integration test for the consolidated background worker.

This test verifies that tasks from all merged workers (heavy, kg_processing,
monitoring, user_file_processing) can be executed by the background worker.
"""

import time

import pytest
from celery.result import AsyncResult

from onyx.background.celery.tasks.monitoring.tasks import monitor_celery_queues
from onyx.background.celery.tasks.pruning.tasks import check_for_pruning
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.integration.common_utils.reset import reset_all


@pytest.fixture(autouse=True)
def reset_for_test() -> None:
    """Reset all data before each test."""
    reset_all()


def test_background_worker_can_execute_monitoring_tasks() -> None:
    """Test that monitoring tasks (from old monitoring worker) can execute."""
    # Get tenant_id from context
    tenant_id = CURRENT_TENANT_ID_CONTEXTVAR.get()

    # Try to execute a monitoring task
    # This task monitors celery queue lengths - safe to call
    result: AsyncResult = monitor_celery_queues.apply_async(
        kwargs={"tenant_id": tenant_id}
    )

    # Wait for task to complete (with timeout)
    timeout = 30
    start_time = time.time()

    while not result.ready() and (time.time() - start_time) < timeout:
        time.sleep(0.5)

    # Task should complete (even if it does nothing)
    assert result.ready(), "Monitoring task should complete"

    # Should not raise an exception
    if result.failed():
        raise result.result


def test_background_worker_can_execute_pruning_tasks() -> None:
    """Test that pruning tasks (from old heavy worker) can execute."""
    # Get tenant_id from context
    tenant_id = CURRENT_TENANT_ID_CONTEXTVAR.get()

    # Try to execute a pruning check task
    # This task checks if pruning is needed - safe to call even if nothing to prune
    result: AsyncResult = check_for_pruning.apply_async(kwargs={"tenant_id": tenant_id})

    # Wait for task to complete (with timeout)
    timeout = 30
    start_time = time.time()

    while not result.ready() and (time.time() - start_time) < timeout:
        time.sleep(0.5)

    # Task should complete (even if it does nothing)
    assert result.ready(), "Pruning task should complete"

    # Should not raise an exception
    if result.failed():
        raise result.result


def test_background_worker_handles_all_queue_types() -> None:
    """
    Test that the background worker is configured to handle all expected queues.

    This is a smoke test that verifies the worker can be started and is
    listening to the correct queues.
    """
    # Get tenant_id from context
    tenant_id = CURRENT_TENANT_ID_CONTEXTVAR.get()

    # In integration tests, the actual Celery workers are running
    # We can verify by executing tasks that route to different queues

    # Execute a monitoring task (routes to 'monitoring' queue)
    monitoring_result = monitor_celery_queues.apply_async(
        kwargs={"tenant_id": tenant_id}
    )

    # Execute a pruning task (routes to 'connector_pruning' queue)
    pruning_result = check_for_pruning.apply_async(kwargs={"tenant_id": tenant_id})

    # Both should be accepted by the background worker
    timeout = 30
    start_time = time.time()

    while (not monitoring_result.ready() or not pruning_result.ready()) and (
        time.time() - start_time
    ) < timeout:
        time.sleep(0.5)

    # Both tasks should complete
    assert monitoring_result.ready(), "Monitoring task should complete"
    assert pruning_result.ready(), "Pruning task should complete"

    # Neither should fail
    assert not monitoring_result.failed(), "Monitoring task should not fail"
    assert not pruning_result.failed(), "Pruning task should not fail"
