import logging
import time
import asyncio
from datetime import datetime

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    # Stub when apscheduler is not installed in local environment
    class BackgroundScheduler:  # type: ignore[no-redef]
        def add_job(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def shutdown(self):
            pass
    class CronTrigger:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

# Configure logging to stdout so Docker captures it
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import ingestion job
try:
    from ingestion import task_runner
    def run_ingestion():
        logger.info("Executing scheduled ingestion job...")
        asyncio.run(task_runner.run_all())
except ImportError:
    def run_ingestion():  # type: ignore[no-redef]
        logger.info("Ingestion job placeholder executed")


def schedule_jobs(scheduler: BackgroundScheduler):
    # Schedule data ingestion every hour
    scheduler.add_job(run_ingestion, CronTrigger(minute=0), id="ingest_hourly", replace_existing=True)
    logger.info("Scheduled ingestion job to run every hour")


def main():
    scheduler = BackgroundScheduler()
    schedule_jobs(scheduler)
    scheduler.start()
    logger.info("APScheduler started. Running jobs...")
    try:
        # Keep the main thread alive without high CPU busy-waiting
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down")


if __name__ == "__main__":
    main()
