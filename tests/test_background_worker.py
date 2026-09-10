import asyncio
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from databases import Database

from comet.background_scraper.worker import (
    BackgroundScraperWorker,
    _serialize_run_row,
)
from comet.core.db_router import ReplicaAwareDatabase
from comet.core.models import settings


async def _create_queue_database(path: Path) -> ReplicaAwareDatabase:
    database = ReplicaAwareDatabase(Database(f"sqlite:///{path}"))
    await database.connect()
    await database.execute(
        """
        CREATE TABLE background_scraper_items (
            media_id TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            next_retry_at REAL,
            last_success_at REAL,
            status TEXT NOT NULL,
            consecutive_failures INTEGER NOT NULL,
            created_at REAL
        )
        """
    )
    await database.execute(
        """
        CREATE TABLE background_scraper_episodes (
            series_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            episode INTEGER NOT NULL,
            next_retry_at REAL,
            last_success_at REAL,
            status TEXT NOT NULL,
            consecutive_failures INTEGER NOT NULL,
            created_at REAL
        )
        """
    )
    return database


class _PassthroughLock:
    def __init__(self, *_args, **_kwargs):
        pass

    async def acquire(self, **_kwargs):
        return True

    async def run(self, task):
        await task

    async def release(self):
        pass


class BackgroundWorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_drain_finishes_active_cycle_without_starting_another(self):
        worker = BackgroundScraperWorker()
        cycle_started = asyncio.Event()
        finish_cycle = asyncio.Event()
        cycle_count = 0

        async def run_cycle():
            nonlocal cycle_count
            cycle_count += 1
            worker.current_run_id = "active-run"
            cycle_started.set()
            await finish_cycle.wait()
            worker.current_run_id = None

        worker._run_scraping_cycle = run_cycle

        with patch("comet.background_scraper.worker.DistributedLock", _PassthroughLock):
            task = asyncio.create_task(worker._run_continuous())
            await cycle_started.wait()

            self.assertTrue(await worker.drain())
            self.assertTrue(await worker.drain())
            self.assertTrue(worker.is_running)
            self.assertTrue(worker._drain_requested)
            worker.is_paused = True
            worker.pause_event.clear()
            worker._discovery_paused_for_backlog = True

            finish_cycle.set()
            await asyncio.wait_for(task, timeout=1)

        self.assertEqual(cycle_count, 1)
        self.assertFalse(worker.is_running)
        self.assertFalse(worker.is_paused)
        self.assertFalse(worker._drain_requested)
        self.assertTrue(worker.pause_event.is_set())
        self.assertFalse(worker._discovery_paused_for_backlog)

    async def test_drain_stops_immediately_when_between_cycles(self):
        worker = BackgroundScraperWorker()
        worker.is_running = True
        worker.stop = AsyncMock()

        self.assertFalse(await worker.drain())

        worker.stop.assert_awaited_once_with()

    async def test_scheduled_stop_can_be_cancelled(self):
        worker = BackgroundScraperWorker()
        worker.is_running = True
        worker.current_run_id = "active-run"

        self.assertTrue(await worker.drain())
        self.assertTrue(worker.cancel_drain())
        self.assertFalse(worker._drain_requested)
        self.assertFalse(worker.cancel_drain())

    async def test_drain_is_honored_when_active_cycle_fails(self):
        worker = BackgroundScraperWorker()
        cycle_started = asyncio.Event()
        fail_cycle = asyncio.Event()

        async def run_cycle():
            worker.current_run_id = "active-run"
            cycle_started.set()
            await fail_cycle.wait()
            worker.current_run_id = None
            raise RuntimeError("cycle failed")

        worker._run_scraping_cycle = run_cycle

        with patch("comet.background_scraper.worker.DistributedLock", _PassthroughLock):
            task = asyncio.create_task(worker._run_continuous())
            await cycle_started.wait()
            self.assertTrue(await worker.drain())
            fail_cycle.set()
            await asyncio.wait_for(task, timeout=1)

        self.assertEqual(worker.last_error, "cycle failed")
        self.assertFalse(worker.is_running)

    async def test_run_insert_failure_clears_published_runtime_state(self):
        worker = BackgroundScraperWorker()
        worker._insert_run_row = AsyncMock(side_effect=RuntimeError("insert failed"))
        worker._reset_running_items = AsyncMock()
        worker._finalize_run_row = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "insert failed"):
            await worker._run_scraping_cycle()

        self.assertIsNone(worker.current_run_id)
        self.assertIsNone(worker.metadata_scraper)
        worker._reset_running_items.assert_not_awaited()
        worker._finalize_run_row.assert_not_awaited()

    async def test_reset_failure_still_finalizes_and_clears_runtime_state(self):
        worker = BackgroundScraperWorker()
        worker._insert_run_row = AsyncMock()
        worker._reset_running_items = AsyncMock(
            side_effect=RuntimeError("reset failed")
        )
        worker._finalize_run_row = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            await worker._run_scraping_cycle()

        worker._finalize_run_row.assert_awaited_once()
        self.assertIsNone(worker.current_run_id)
        self.assertIsNone(worker.metadata_scraper)

    async def test_finalize_failure_clears_runtime_state(self):
        worker = BackgroundScraperWorker()
        worker._insert_run_row = AsyncMock()
        worker._reset_running_items = AsyncMock()
        worker._finalize_run_row = AsyncMock(
            side_effect=RuntimeError("finalize failed")
        )

        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            await worker._run_scraping_cycle()

        self.assertIsNone(worker.current_run_id)
        self.assertIsNone(worker.metadata_scraper)

    async def test_continuous_runner_propagates_cancellation(self):
        worker = BackgroundScraperWorker()
        entered_sleep = asyncio.Event()

        async def blocked_sleep(_delay):
            entered_sleep.set()
            await asyncio.Event().wait()

        with (
            patch(
                "comet.background_scraper.worker.DistributedLock.acquire",
                new=AsyncMock(return_value=False),
            ),
            patch("comet.background_scraper.worker.asyncio.sleep", new=blocked_sleep),
        ):
            task = asyncio.create_task(worker._run_continuous())
            await entered_sleep.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertFalse(worker.is_running)


class BackgroundWorkerQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_snapshot_query_executes_against_sqlite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "queue.db")
            try:
                await database.execute_many(
                    """
                    INSERT INTO background_scraper_items
                    (media_id, media_type, status, consecutive_failures, created_at)
                    VALUES (:media_id, :media_type, 'discovered', 0, :created_at)
                    """,
                    [
                        {
                            "media_id": "movie",
                            "media_type": "movie",
                            "created_at": 90.0,
                        },
                        {
                            "media_id": "series",
                            "media_type": "series",
                            "created_at": 85.0,
                        },
                    ],
                )
                await database.execute(
                    """
                    INSERT INTO background_scraper_episodes
                    (series_id, season, episode, status, consecutive_failures, created_at)
                    VALUES ('series', 1, 1, 'discovered', 0, 80.0)
                    """
                )

                with patch("comet.background_scraper.worker.database", database):
                    snapshot = await BackgroundScraperWorker()._fetch_queue_snapshot(
                        now=100.0
                    )
            finally:
                await database.disconnect()

        self.assertEqual(snapshot["total"], 3)
        self.assertEqual(snapshot["oldest_age_s"], 15.0)

    async def test_queue_age_starts_when_success_becomes_refresh_eligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "queue.db")
            try:
                await database.execute(
                    """
                    INSERT INTO background_scraper_items
                    (media_id, media_type, next_retry_at, last_success_at, status,
                     consecutive_failures, created_at)
                    VALUES ('movie', 'movie', 97.0, 89.0, 'success', 0, 1.0)
                    """
                )

                with (
                    patch.object(settings, "BACKGROUND_SCRAPER_SUCCESS_TTL", 10),
                    patch("comet.background_scraper.worker.database", database),
                ):
                    snapshot = await BackgroundScraperWorker()._fetch_queue_snapshot(
                        now=100.0
                    )
            finally:
                await database.disconnect()

        self.assertEqual(snapshot["movies"], 1)
        self.assertEqual(snapshot["oldest_age_s"], 1.0)

    async def test_queue_age_starts_when_retry_becomes_eligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "queue.db")
            try:
                await database.execute(
                    """
                    INSERT INTO background_scraper_items
                    (media_id, media_type, next_retry_at, last_success_at, status,
                     consecutive_failures, created_at)
                    VALUES ('movie', 'movie', 98.0, NULL, 'failed', 1, 1.0)
                    """
                )

                with patch("comet.background_scraper.worker.database", database):
                    snapshot = await BackgroundScraperWorker()._fetch_queue_snapshot(
                        now=100.0
                    )
            finally:
                await database.disconnect()

        self.assertEqual(snapshot["movies"], 1)
        self.assertEqual(snapshot["oldest_age_s"], 2.0)

    async def test_episode_queue_age_waits_for_parent_series(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "queue.db")
            try:
                await database.execute(
                    """
                    INSERT INTO background_scraper_items
                    (media_id, media_type, next_retry_at, last_success_at, status,
                     consecutive_failures, created_at)
                    VALUES ('series', 'series', 99.0, NULL, 'deferred', 0, 1.0)
                    """
                )
                await database.execute(
                    """
                    INSERT INTO background_scraper_episodes
                    (series_id, season, episode, next_retry_at, last_success_at,
                     status, consecutive_failures, created_at)
                    VALUES ('series', 1, 1, 90.0, NULL, 'failed', 1, 1.0)
                    """
                )

                with patch("comet.background_scraper.worker.database", database):
                    snapshot = await BackgroundScraperWorker()._fetch_queue_snapshot(
                        now=100.0
                    )
            finally:
                await database.disconnect()

        self.assertEqual(snapshot["series"], 1)
        self.assertEqual(snapshot["episodes"], 1)
        self.assertEqual(snapshot["oldest_age_s"], 1.0)

    async def test_queue_snapshot_uses_one_primary_database_snapshot(self):
        worker = BackgroundScraperWorker()
        fetch_one = AsyncMock(
            return_value={
                "movie_count": 2,
                "series_count": 3,
                "oldest_item_ts": 90.0,
                "episode_count": 5,
                "oldest_episode_ts": 80.0,
            }
        )

        with patch("comet.background_scraper.worker.database.fetch_one", fetch_one):
            snapshot = await worker._fetch_queue_snapshot(now=100.0)

        self.assertEqual(
            snapshot,
            {
                "movies": 2,
                "series": 3,
                "episodes": 5,
                "total": 10,
                "oldest_age_s": 20.0,
            },
        )
        fetch_one.assert_awaited_once()
        self.assertTrue(fetch_one.await_args.kwargs["force_primary"])
        self.assertIn("CROSS JOIN episode_snapshot", fetch_one.await_args.args[0])

    async def test_queue_snapshot_rejects_corrupt_database_values(self):
        worker = BackgroundScraperWorker()
        valid = {
            "movie_count": 2,
            "series_count": 3,
            "oldest_item_ts": 90.0,
            "episode_count": 5,
            "oldest_episode_ts": 80.0,
        }
        invalid_rows = [
            None,
            valid | {"extra": 1},
            valid | {"movie_count": True},
            valid | {"series_count": -1},
            valid | {"episode_count": 1.5},
            valid | {"oldest_item_ts": 90},
            valid | {"oldest_item_ts": math.nan},
            valid | {"oldest_episode_ts": math.inf},
        ]

        for row in invalid_rows:
            with (
                self.subTest(row=row),
                patch(
                    "comet.background_scraper.worker.database.fetch_one",
                    new=AsyncMock(return_value=row),
                ),
                self.assertRaises((TypeError, ValueError)),
            ):
                await worker._fetch_queue_snapshot(now=100.0)

    async def test_requeue_dead_items_rolls_back_both_tables_on_failure(self):
        worker = BackgroundScraperWorker()

        class Transaction:
            def __init__(self):
                self.exit_error = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, error_type, error, traceback):
                self.exit_error = error

        transaction = Transaction()
        fetch_val = AsyncMock(side_effect=[2, 3])
        execute = AsyncMock(side_effect=[None, RuntimeError("episode update failed")])

        with (
            patch(
                "comet.background_scraper.worker.database.transaction",
                return_value=transaction,
            ),
            patch("comet.background_scraper.worker.database.fetch_val", fetch_val),
            patch("comet.background_scraper.worker.database.execute", execute),
            self.assertRaisesRegex(RuntimeError, "episode update failed"),
        ):
            await worker.requeue_dead_items()

        self.assertIsInstance(transaction.exit_error, RuntimeError)
        self.assertEqual(fetch_val.await_count, 2)
        self.assertEqual(execute.await_count, 2)

    async def test_recent_runs_enforces_limit_and_serializes_current_schema(self):
        worker = BackgroundScraperWorker()
        row = {
            "run_id": "12345678-1234-4234-8234-123456789abc",
            "started_at": 10.0,
            "finished_at": 11.0,
            "status": "completed",
            "processed": 3,
            "success": 2,
            "failed": 1,
            "torrents_found": 4,
            "duration_ms": 1000,
            "worker_count": 2,
            "last_error": None,
        }
        fetch_all = AsyncMock(return_value=[row])

        with patch("comet.background_scraper.worker.database.fetch_all", fetch_all):
            self.assertEqual(await worker.get_recent_runs(20), [row])

        for limit in (None, True, 0, -1, 201, 1.5, "20"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                await worker.get_recent_runs(limit)

        fetch_all.assert_awaited_once()

    def test_run_rows_reject_corrupt_schema_and_invariants(self):
        valid = {
            "run_id": "12345678-1234-4234-8234-123456789abc",
            "started_at": 10.0,
            "finished_at": 11.0,
            "status": "completed",
            "processed": 3,
            "success": 2,
            "failed": 1,
            "torrents_found": 4,
            "duration_ms": 1000,
            "worker_count": 2,
            "last_error": None,
        }
        invalid_rows = [
            None,
            valid | {"extra": 1},
            valid | {"run_id": "not-a-uuid"},
            valid | {"status": "dead"},
            valid | {"started_at": 10},
            valid | {"started_at": math.nan},
            valid | {"finished_at": 9.0},
            valid | {"finished_at": None},
            valid | {"processed": True},
            valid | {"failed": -1},
            valid | {"duration_ms": 1.5},
            valid | {"last_error": 1},
        ]

        self.assertEqual(_serialize_run_row(valid), valid)
        self.assertIsNone(
            _serialize_run_row(valid | {"status": "running", "finished_at": None})[
                "finished_at"
            ]
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises((TypeError, ValueError)):
                _serialize_run_row(row)


class BackgroundWorkerPhase1Tests(unittest.IsolatedAsyncioTestCase):
    async def test_get_blocked_candidate_ids_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "queue.db")
            try:
                await database.execute_many(
                    """
                    INSERT INTO background_scraper_items
                    (media_id, media_type, next_retry_at, last_success_at, status, consecutive_failures, created_at)
                    VALUES (:media_id, :media_type, :next_retry_at, :last_success_at, :status, :consecutive_failures, :created_at)
                    """,
                    [
                        {
                            "media_id": "tt_blocked_retry",
                            "media_type": "movie",
                            "next_retry_at": 150.0,
                            "last_success_at": None,
                            "status": "deferred",
                            "consecutive_failures": 0,
                            "created_at": 10.0,
                        },
                        {
                            "media_id": "tt_blocked_success",
                            "media_type": "movie",
                            "next_retry_at": None,
                            "last_success_at": 95.0,
                            "status": "success",
                            "consecutive_failures": 0,
                            "created_at": 10.0,
                        },
                        {
                            "media_id": "tt_ready_retry",
                            "media_type": "movie",
                            "next_retry_at": 80.0,
                            "last_success_at": None,
                            "status": "failed",
                            "consecutive_failures": 1,
                            "created_at": 10.0,
                        },
                    ],
                )

                worker = BackgroundScraperWorker()
                with patch("comet.background_scraper.worker.database", database):
                    blocked = await worker._get_blocked_candidate_ids(
                        ["tt_blocked_retry", "tt_blocked_success", "tt_ready_retry", "tt_nonexistent"],
                        media_type="movie",
                        now=100.0,
                        success_cutoff=90.0,
                    )
                    self.assertIn("tt_blocked_retry", blocked)
                    self.assertIn("tt_blocked_success", blocked)
                    self.assertNotIn("tt_ready_retry", blocked)
                    self.assertNotIn("tt_nonexistent", blocked)

                    # Also test single-item wrapper
                    self.assertTrue(
                        await worker._is_discovery_candidate_blocked(
                            "tt_blocked_retry", "movie", now=100.0, success_cutoff=90.0
                        )
                    )
                    self.assertFalse(
                        await worker._is_discovery_candidate_blocked(
                            "tt_ready_retry", "movie", now=100.0, success_cutoff=90.0
                        )
                    )
            finally:
                await database.disconnect()

    async def test_plan_items_sargable_demand_matching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "plan.db")
            try:
                await database.execute(
                    """
                    CREATE TABLE media_demand (
                        media_id TEXT PRIMARY KEY,
                        last_seen_at REAL NOT NULL
                    )
                    """
                )
                await database.execute(
                    "ALTER TABLE background_scraper_items ADD COLUMN title TEXT NOT NULL DEFAULT ''"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_items ADD COLUMN year INTEGER NOT NULL DEFAULT 2024"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_items ADD COLUMN year_end INTEGER"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_items ADD COLUMN priority_score REAL NOT NULL DEFAULT 10.0"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_items ADD COLUMN last_scraped_at REAL"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_items ADD COLUMN updated_at REAL"
                )

                # Insert items
                await database.execute_many(
                    """
                    INSERT INTO background_scraper_items
                    (media_id, media_type, title, priority_score, status, consecutive_failures, created_at)
                    VALUES (:media_id, :media_type, :title, :priority_score, 'discovered', 0, 10.0)
                    """,
                    [
                        {"media_id": "tt_series_demand", "media_type": "series", "title": "Series Demand", "priority_score": 10.0},
                        {"media_id": "tt_series_normal", "media_type": "series", "title": "Series Normal", "priority_score": 10.0},
                        {"media_id": "tt_movie_demand", "media_type": "movie", "title": "Movie Demand", "priority_score": 5.0},
                    ],
                )

                # Insert demand: series demand has episode suffix, movie demand has root id
                now = time.time()
                await database.execute_many(
                    """
                    INSERT INTO media_demand (media_id, last_seen_at)
                    VALUES (:media_id, :last_seen_at)
                    """,
                    [
                        {"media_id": "tt_series_demand:1:1", "last_seen_at": now - 5.0},
                        {"media_id": "tt_movie_demand", "last_seen_at": now - 5.0},
                    ],
                )

                worker = BackgroundScraperWorker()
                with (
                    patch("comet.background_scraper.worker.database", database),
                    patch.object(settings, "BACKGROUND_SCRAPER_ENABLE_DEMAND_PRIORITY", True),
                    patch.object(settings, "BACKGROUND_SCRAPER_DEMAND_LOOKBACK", 50),
                    patch.object(settings, "BACKGROUND_SCRAPER_SUCCESS_TTL", 604800),
                ):
                    series_plan = await worker._plan_items("series", 10)
                    self.assertEqual(len(series_plan), 2)
                    # The series with demand boost (+100) should be first
                    self.assertEqual(series_plan[0]["media_id"], "tt_series_demand")
                    self.assertEqual(series_plan[0]["demand_boost"], 100.0)
                    self.assertEqual(series_plan[1]["media_id"], "tt_series_normal")
                    self.assertEqual(series_plan[1]["demand_boost"], 0.0)

                    movie_plan = await worker._plan_items("movie", 10)
                    self.assertEqual(len(movie_plan), 1)
                    self.assertEqual(movie_plan[0]["media_id"], "tt_movie_demand")
                    self.assertEqual(movie_plan[0]["demand_boost"], 100.0)
            finally:
                await database.disconnect()

    async def test_live_traffic_deference_pauses_worker(self):
        worker = BackgroundScraperWorker()
        worker.is_running = True

        sleep_calls = []
        async def fake_sleep(dur):
            sleep_calls.append(dur)
            worker.is_running = False  # break out of loop after first wait

        with (
            patch("comet.background_scraper.worker.get_last_live_scrape_time", return_value=100.0),
            patch("time.time", return_value=110.0),  # 10s elapsed
            patch.object(settings, "BACKGROUND_SCRAPER_LIVE_DEFERENCE_SECONDS", 45),
            patch("asyncio.sleep", fake_sleep),
        ):
            await worker._wait_if_live_traffic_active()

        # Should have waited since elapsed (10s) < deference (45s)
        self.assertTrue(len(sleep_calls) > 0)
        self.assertAlmostEqual(sleep_calls[0], 5.0)

    async def test_batch_update_episodes_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = await _create_queue_database(Path(temp_dir) / "episodes.db")
            try:
                await database.execute(
                    "ALTER TABLE background_scraper_episodes ADD COLUMN episode_media_id TEXT"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_episodes ADD COLUMN last_scraped_at REAL"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_episodes ADD COLUMN last_failure_at REAL"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_episodes ADD COLUMN total_torrents_found INTEGER DEFAULT 0"
                )
                await database.execute(
                    "ALTER TABLE background_scraper_episodes ADD COLUMN updated_at REAL"
                )

                await database.execute_many(
                    """
                    INSERT INTO background_scraper_episodes
                    (episode_media_id, series_id, season, episode, status, consecutive_failures, created_at)
                    VALUES (:episode_media_id, :series_id, :season, :episode, 'discovered', 1, 10.0)
                    """,
                    [
                        {"episode_media_id": "tt100:1:1", "series_id": "tt100", "season": 1, "episode": 1},
                        {"episode_media_id": "tt100:1:2", "series_id": "tt100", "season": 1, "episode": 2},
                    ],
                )

                worker = BackgroundScraperWorker()
                with (
                    patch("comet.background_scraper.worker.database", database),
                    patch("comet.background_scraper.worker.mark_scope_scraped", AsyncMock()) as mock_mark,
                ):
                    episodes = [
                        {"episode_media_id": "tt100:1:1"},
                        {"episode_media_id": "tt100:1:2"},
                    ]
                    await worker._batch_update_episodes_success(episodes, torrents_found=8)

                    rows = await database.fetch_all(
                        "SELECT episode_media_id, status, consecutive_failures, total_torrents_found FROM background_scraper_episodes"
                    )
                    self.assertEqual(len(rows), 2)
                    for r in rows:
                        self.assertEqual(r["status"], "success")
                        self.assertEqual(r["consecutive_failures"], 0)
                        self.assertEqual(r["total_torrents_found"], 8)

                    self.assertEqual(mock_mark.await_count, 2)
            finally:
                await database.disconnect()

    async def test_scrape_series_season_pack_first_auto_resolves(self):
        worker = BackgroundScraperWorker()
        worker.is_running = True
        worker.metadata_scraper = AsyncMock()
        worker.metadata_scraper.fetch_aliases_with_metadata = AsyncMock(
            return_value=({"title": "Test Show", "year": 2024, "year_end": None}, {})
        )

        episodes = [
            {"episode_media_id": "tt999:1:1", "series_id": "tt999", "season": 1, "episode": 1, "consecutive_failures": 0},
            {"episode_media_id": "tt999:1:2", "series_id": "tt999", "season": 1, "episode": 2, "consecutive_failures": 0},
        ]
        worker._get_or_discover_episodes = AsyncMock(return_value=episodes)
        worker._batch_update_episodes_success = AsyncMock()
        worker._update_episode_state = AsyncMock()

        class FakeParsed:
            seasons = [1]
            episodes = []

        class FakeTorrentManager:
            def __init__(self, *args, **kwargs):
                self.season = kwargs.get("season")
                self.episode = kwargs.get("episode")
                self.torrents = {
                    "hash_season_pack": {
                        "parsed": FakeParsed(),
                        "title": "Test Show S01 1080p Complete",
                    }
                }

            async def scrape_torrents(self, context):
                pass

        with (
            patch("comet.background_scraper.worker.TorrentManager", FakeTorrentManager),
            patch.object(settings, "BACKGROUND_SCRAPER_SEASON_PACK_FIRST", True),
        ):
            item = {"media_id": "tt999", "title": "Test Show", "year": 2024, "year_end": None}
            total = await worker._scrape_series(item, deadline=None)

            self.assertEqual(total, 1)
            # Batch update was called for both episodes at once!
            worker._batch_update_episodes_success.assert_awaited_once_with(episodes, 1)
            # Individual episode state update was NOT needed because all were auto-resolved!
            worker._update_episode_state.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
