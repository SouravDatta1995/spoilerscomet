import asyncio
import aiohttp
import re

from comet.core.constants import INDEXER_TIMEOUT
from comet.core.logger import logger
from comet.core.models import settings
from comet.scrapers.base import (
    BaseScraper,
    deduplicate_torrents,
    gather_with_error_logging,
)
from comet.scrapers.models import ScrapeRequest, ScrapeResult
from comet.services.indexer_manager import indexer_manager
from comet.services.torrent_manager import (
    add_torrent_queue,
    download_torrent,
    extract_torrent_metadata,
    extract_trackers_from_magnet,
)

# Slow, Cloudflare-protected, or non-magnet indexers to exclude ONLY during live searches
# (These are 100% enabled and used by the background scraper)
SLOW_INDEXER_PATTERNS = [r"limetorrents", r"torrentproject", r"yts", r"1337x", r"1337"]

class ProwlarrScraper(BaseScraper):
    def __init__(self, manager, session, url: str):
        super().__init__(manager, session, url)
        self._indexer_cache = {}

    async def _get_indexer_info(self):
        try:
            async with self.session.get(
                f"{self.url}/api/v1/indexer",
                headers={"X-Api-Key": settings.PROWLARR_API_KEY},
                timeout=5.0,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._indexer_cache = {
                        str(idx["id"]): idx.get("name", "").lower()
                        for idx in data
                        if isinstance(idx, dict) and "id" in idx
                    }
        except Exception:
            pass

    async def process_torrent(self, result: dict, media_id: str, season: int):
        base_torrent = {
            "title": result["title"],
            "infoHash": None,
            "fileIndex": None,
            "seeders": int(result["seeders"])
            if result["seeders"] is not None
            else None,
            "size": result["size"],
            "tracker": result["indexer"],
            "sources": [],
        }

        torrents = []

        if "downloadUrl" in result and settings.DOWNLOAD_TORRENT_FILES:
            download_url = result["downloadUrl"]
            if self.url:
                download_url = re.sub(r"^https?://(localhost|127\.0\.0\.1):9696", self.url.rstrip("/"), download_url)
            content, magnet_hash, magnet_url = await download_torrent(
                self.session, download_url
            )

            if content:
                metadata = await asyncio.to_thread(extract_torrent_metadata, content)
                if metadata:
                    for file in metadata["files"]:
                        torrent = base_torrent.copy()
                        torrent["title"] = file["title"]
                        torrent["infoHash"] = metadata["info_hash"].lower()
                        torrent["fileIndex"] = file["index"]
                        torrent["size"] = file["size"]
                        torrent["sources"] = metadata["sources"]
                        torrents.append(torrent)
                    return torrents

            if magnet_hash:
                base_torrent["infoHash"] = magnet_hash.lower()
                base_torrent["sources"] = extract_trackers_from_magnet(magnet_url)

                await add_torrent_queue.add_torrent(
                    magnet_url,
                    base_torrent["seeders"],
                    base_torrent["tracker"],
                    media_id,
                    season,
                    base_torrent["infoHash"],
                )

                torrents.append(base_torrent)
                return torrents

        if result.get("infoHash"):
            base_torrent["infoHash"] = result["infoHash"].lower()
            if "guid" in result and result["guid"].startswith("magnet:"):
                base_torrent["sources"] = extract_trackers_from_magnet(result["guid"])

                await add_torrent_queue.add_torrent(
                    result["guid"],
                    base_torrent["seeders"],
                    base_torrent["tracker"],
                    media_id,
                    season,
                    base_torrent["infoHash"],
                )

            torrents.append(base_torrent)

        return torrents

    async def _fetch_search_results(self, query, indexer_ids, is_live=True):
        url = f"{self.url}/api/v1/search"
        params = [
            ("query", query),
            *(("indexerIds", indexer_id) for indexer_id in indexer_ids),
            ("type", "search"),
        ]
        # 3.5s timeout for live searches so TorrentProject2 (2.9s) returns without hitting AIOStreams 10s limit; 45s for background
        timeout_val = aiohttp.ClientTimeout(total=3.5) if is_live else aiohttp.ClientTimeout(total=45)
        async with self.session.get(
            url,
            params=params,
            headers={"X-Api-Key": settings.PROWLARR_API_KEY},
            timeout=timeout_val,
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            data = await response.json()
            if not isinstance(data, list):
                raise ValueError("response payload is not a list")
            return data

    async def scrape(self, request: ScrapeRequest):
        if not settings.PROWLARR_INDEXERS:
            try:
                await asyncio.wait_for(
                    indexer_manager.prowlarr_initialized.wait(),
                    timeout=settings.INDEXER_MANAGER_WAIT_TIMEOUT,
                )
            except TimeoutError:
                logger.warning(
                    "Timed out waiting for Prowlarr indexers; skipping scrape."
                )
                return []

        if not settings.PROWLARR_INDEXERS:
            logger.warning("No Prowlarr indexers available, skipping scrape.")
            return []

        # Update indexer name cache if empty
        if not self._indexer_cache:
            await self._get_indexer_info()

        is_live = getattr(request.context, "value", str(request.context)).lower() == "live"
        target_indexers = settings.PROWLARR_INDEXERS
        if is_live:
            target_indexers = [
                i
                for i in target_indexers
                if not any(
                    re.search(pat, self._indexer_cache.get(str(i), "").lower())
                    for pat in SLOW_INDEXER_PATTERNS
                )
            ]
        active_names = [self._indexer_cache.get(str(i), str(i)) for i in target_indexers]
        logger.log("SCRAPER", f"Prowlarr search ({'live' if is_live else 'background'}) using {len(target_indexers)} indexers: {', '.join(active_names)}")

        torrents: list[ScrapeResult] = []
        seen: set[str] = set()

        if is_live:
            # For live search, use primary clean title so Prowlarr finishes in < 1.5s
            clean_title = re.sub(r'[:\-–—].*$', '', request.title).strip()
            queries = [clean_title]
        else:
            queries = request.title_queries(include_episode_variants=True)

        try:
            responses = await gather_with_error_logging(
                (
                    f"Prowlarr query {query!r} ({self.url})",
                    self._fetch_search_results(query, target_indexers, is_live=is_live),
                )
                for query in queries
            )
            all_results = []
            for response in responses:
                all_results.extend(response)

            torrent_tasks = []
            for result in all_results:
                if not isinstance(result, dict):
                    continue
                info_url = result.get("infoUrl")
                if not isinstance(info_url, str) or not info_url or info_url in seen:
                    continue

                seen.add(info_url)
                torrent_tasks.append(
                    (
                        f"Prowlarr result {info_url!r}",
                        self.process_torrent(
                            result, request.media_only_id, request.season
                        ),
                    )
                )

            processed_torrents = await gather_with_error_logging(torrent_tasks)
            for sublist in processed_torrents:
                for t in sublist:
                    if isinstance(t, dict) and t.get("infoHash"):
                        torrents.append(t)

        except Exception as e:
            logger.warning(
                f"Exception while getting torrents for {request.title} with Prowlarr: {e}"
            )

        return deduplicate_torrents(torrents)
