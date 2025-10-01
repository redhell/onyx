import subprocess
import threading


def monitor_process(process_name: str, process: subprocess.Popen) -> None:
    assert process.stdout is not None

    while True:
        output = process.stdout.readline()

        if output:
            print(f"{process_name}: {output.strip()}")

        if process.poll() is not None:
            break


def run_jobs() -> None:
    # command setup
    cmd_worker_primary = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.primary",
        "worker",
        "--pool=threads",
        "--concurrency=6",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=primary@%n",
        "-Q",
        "celery",
    ]

    cmd_worker_light = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.light",
        "worker",
        "--pool=threads",
        "--concurrency=16",
        "--prefetch-multiplier=8",
        "--loglevel=INFO",
        "--hostname=light@%n",
        "-Q",
        "vespa_metadata_sync,connector_deletion,doc_permissions_upsert,checkpoint_cleanup,index_attempt_cleanup",
    ]

    cmd_worker_background = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.background",
        "worker",
        "--pool=threads",
        "--concurrency=6",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=background@%n",
        "-Q",
        "connector_pruning,connector_doc_permissions_sync,connector_external_group_sync,csv_generation,kg_processing,monitoring,user_file_processing,user_file_project_sync",
    ]

    cmd_worker_docprocessing = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.docprocessing",
        "worker",
        "--pool=threads",
        "--concurrency=6",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=docprocessing@%n",
        "--queues=docprocessing",
    ]

    cmd_worker_docfetching = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.docfetching",
        "worker",
        "--pool=threads",
        "--concurrency=1",
        "--prefetch-multiplier=1",
        "--loglevel=INFO",
        "--hostname=docfetching@%n",
        "--queues=connector_doc_fetching,user_files_indexing",
    ]

    cmd_beat = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.beat",
        "beat",
        "--loglevel=INFO",
    ]

    # spawn processes
    worker_primary_process = subprocess.Popen(
        cmd_worker_primary, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    worker_light_process = subprocess.Popen(
        cmd_worker_light, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    worker_background_process = subprocess.Popen(
        cmd_worker_background,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    worker_docprocessing_process = subprocess.Popen(
        cmd_worker_docprocessing,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    worker_docfetching_process = subprocess.Popen(
        cmd_worker_docfetching,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    beat_process = subprocess.Popen(
        cmd_beat, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # monitor threads
    worker_primary_thread = threading.Thread(
        target=monitor_process, args=("PRIMARY", worker_primary_process)
    )
    worker_light_thread = threading.Thread(
        target=monitor_process, args=("LIGHT", worker_light_process)
    )
    worker_background_thread = threading.Thread(
        target=monitor_process, args=("BACKGROUND", worker_background_process)
    )
    worker_docprocessing_thread = threading.Thread(
        target=monitor_process, args=("DOCPROCESSING", worker_docprocessing_process)
    )
    worker_docfetching_thread = threading.Thread(
        target=monitor_process, args=("DOCFETCHING", worker_docfetching_process)
    )
    beat_thread = threading.Thread(target=monitor_process, args=("BEAT", beat_process))

    worker_primary_thread.start()
    worker_light_thread.start()
    worker_background_thread.start()
    worker_docprocessing_thread.start()
    worker_docfetching_thread.start()
    beat_thread.start()

    worker_primary_thread.join()
    worker_light_thread.join()
    worker_background_thread.join()
    worker_docprocessing_thread.join()
    worker_docfetching_thread.join()
    beat_thread.join()


if __name__ == "__main__":
    run_jobs()
