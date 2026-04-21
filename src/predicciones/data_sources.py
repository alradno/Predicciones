from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import pandas as pd
import requests

try:  # pragma: no cover - depende de extra opcional
    import websockets
except Exception:  # pragma: no cover - el collector sigue funcionando por REST
    websockets = None


FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_NEW_URL = "https://www.football-data.co.uk/new"
POLYMARKET_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_BASE_URL = "https://clob.polymarket.com"

LEAGUE_NAMES: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "MEX": "Liga MX",
    "USA": "MLS",
}

NEW_FOOTBALL_DATA_LEAGUES: set[str] = {"MEX", "USA"}


def normalize_season_code(season: str | int) -> str:
    value = str(season).strip().replace("/", "")
    if len(value) == 4 and value.isdigit():
        return value
    raise ValueError(f"Temporada no valida: {season!r}. Usa formato 2425, 2324, etc.")


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


class HistoricalMatchProvider(ABC):
    @abstractmethod
    def download(self, leagues: list[str] | tuple[str, ...], seasons: list[str] | tuple[str, ...]) -> "DownloadResult":
        raise NotImplementedError


@dataclass(frozen=True)
class DownloadResult:
    matches: pd.DataFrame
    failures: list[str]


class FootballDataClient(HistoricalMatchProvider):
    """Cliente sencillo para football-data.co.uk."""

    keep_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
        "B365H",
        "B365D",
        "B365A",
    ]

    @staticmethod
    def _season_labels_for_extra_file(season_code: str) -> set[str]:
        if season_code.startswith("20"):
            end_year = int(season_code)
            start_year = end_year - 1
        else:
            start_year = 2000 + int(season_code[:2])
            end_year = 2000 + int(season_code[2:])
            if end_year < start_year:
                end_year += 100
        return {f"{start_year}/{end_year}", str(end_year)}

    def _load_extra_league(self, league_code: str, season_code: str) -> pd.DataFrame:
        url = f"{FOOTBALL_DATA_NEW_URL}/{league_code}.csv"
        df = pd.read_csv(url, encoding="utf-8-sig", on_bad_lines="skip")
        labels = self._season_labels_for_extra_file(season_code)
        if "Season" in df.columns:
            df = df[df["Season"].astype(str).isin(labels)].copy()
        rename_map = {
            "Home": "HomeTeam",
            "Away": "AwayTeam",
            "HG": "FTHG",
            "AG": "FTAG",
            "Res": "FTR",
            "B365CH": "B365H",
            "B365CD": "B365D",
            "B365CA": "B365A",
        }
        df = df.rename(columns={source: target for source, target in rename_map.items() if source in df.columns})
        available = [column for column in self.keep_columns if column in df.columns]
        trimmed = df[available].copy()
        trimmed["league_code"] = league_code
        trimmed["league_name"] = LEAGUE_NAMES.get(league_code, league_code)
        trimmed["season"] = season_code
        trimmed["source_url"] = url
        return trimmed

    def load_one(self, league_code: str, season: str) -> pd.DataFrame:
        season_code = normalize_season_code(season)
        if league_code in NEW_FOOTBALL_DATA_LEAGUES:
            return self._load_extra_league(league_code, season_code)

        url = f"{FOOTBALL_DATA_BASE_URL}/{season_code}/{league_code}.csv"

        df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
        available = [column for column in self.keep_columns if column in df.columns]
        trimmed = df[available].copy()
        trimmed["league_code"] = league_code
        trimmed["league_name"] = LEAGUE_NAMES.get(league_code, league_code)
        trimmed["season"] = season_code
        trimmed["source_url"] = url
        return trimmed

    def download(self, leagues: list[str] | tuple[str, ...], seasons: list[str] | tuple[str, ...]) -> DownloadResult:
        frames: list[pd.DataFrame] = []
        failures: list[str] = []

        for league in leagues:
            for season in seasons:
                try:
                    frame = self.load_one(league, season)
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:  # pragma: no cover - depende de red
                    failures.append(f"{league}/{season}: {exc}")

        if not frames:
            raise RuntimeError("No se pudo descargar ninguna temporada.")

        combined = pd.concat(frames, ignore_index=True)
        return DownloadResult(matches=combined, failures=failures)


class PolymarketGammaClient:
    """Wrapper sobre la Gamma API publica."""

    def __init__(self, base_url: str = POLYMARKET_GAMMA_BASE_URL, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "predicciones-football/0.3"})

    def _get(self, path: str, **params: Any) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_markets(self, limit: int = 100, **params: Any) -> list[dict[str, Any]]:
        payload = self._get("/markets", limit=limit, **params)
        return payload if isinstance(payload, list) else []

    def list_events(self, limit: int = 100, **params: Any) -> list[dict[str, Any]]:
        payload = self._get("/events", limit=limit, **params)
        return payload if isinstance(payload, list) else []

    def list_events_keyset(
        self,
        limit: int = 100,
        after_cursor: str | None = None,
        **params: Any,
    ) -> tuple[list[dict[str, Any]], str | None]:
        payload = self._get("/events/keyset", limit=limit, after_cursor=after_cursor, **params)
        if not isinstance(payload, dict):
            return [], None
        events = payload.get("events", [])
        next_cursor = payload.get("next_cursor")
        return (events if isinstance(events, list) else []), (str(next_cursor) if next_cursor else None)

    def list_sports(self) -> list[dict[str, Any]]:
        payload = self._get("/sports")
        return payload if isinstance(payload, list) else []

    def search_events(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        payload = self._get("/public-search", q=query)
        events = payload.get("events", [])
        return events[:limit]

    def search_markets(self, query: str, limit: int = 10, active: bool = True, closed: bool = False) -> list[dict[str, Any]]:
        events = self.search_events(query=query, limit=limit)
        collected: list[dict[str, Any]] = []

        for event in events:
            for market in event.get("markets", []):
                item = dict(market)
                item["eventSlug"] = event.get("slug")
                item["eventTitle"] = event.get("title")

                if active and not item.get("active", False):
                    continue
                if not closed and item.get("closed", False):
                    continue

                collected.append(item)
                if len(collected) >= limit:
                    return collected

        return collected[:limit]

    def event_by_slug(self, slug: str) -> list[dict[str, Any]]:
        return self._get("/events", slug=slug)

    def market_by_id(self, market_id: str | int) -> dict[str, Any] | None:
        payload = self._get("/markets", id=market_id)
        if isinstance(payload, list):
            return payload[0] if payload else None
        if isinstance(payload, dict):
            return payload
        return None

    @staticmethod
    def parse_outcomes(market: dict[str, Any]) -> list[str]:
        return [str(item) for item in _json_list(market.get("outcomes"))]

    @staticmethod
    def parse_clob_token_ids(market: dict[str, Any]) -> list[str]:
        return [str(item) for item in _json_list(market.get("clobTokenIds"))]

    @staticmethod
    def extract_yes_price(market: dict[str, Any]) -> float | None:
        prices = _json_list(market.get("outcomePrices"))
        outcomes = PolymarketGammaClient.parse_outcomes(market)
        if outcomes and prices:
            for outcome, price in zip(outcomes, prices):
                if str(outcome).strip().lower() == "yes":
                    return float(price)
        if prices:
            return float(prices[0])
        return None

    @staticmethod
    def extract_fee_rate(market: dict[str, Any]) -> float:
        schedule = market.get("feeSchedule") or {}
        if isinstance(schedule, dict) and "rate" in schedule:
            return float(schedule["rate"])
        if market.get("takerBaseFee") is not None:
            return float(market["takerBaseFee"]) / 10000.0
        return 0.0


class PolymarketClobClient:
    """Cliente REST/WebSocket para el CLOB oficial de Polymarket."""

    def __init__(self, base_url: str = POLYMARKET_CLOB_BASE_URL, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "predicciones-football/0.3"})

    def _get(self, path: str, **params: Any) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_order_book(self, token_id: str | int) -> dict[str, Any]:
        return self._get("/book", token_id=str(token_id))

    def get_midpoint(self, token_id: str | int) -> dict[str, Any]:
        return self._get("/midpoint", token_id=str(token_id))

    def get_spread(self, token_id: str | int) -> dict[str, Any]:
        return self._get("/spread", token_id=str(token_id))

    def get_last_trade_price(self, token_id: str | int) -> dict[str, Any]:
        return self._get("/last-trade-price", token_id=str(token_id))

    def get_fee_rate(self, token_id: str | int) -> dict[str, Any]:
        return self._get("/fee-rate", token_id=str(token_id))

    def get_prices_history(
        self,
        market_id: str | int,
        interval: str = "1h",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/prices-history",
            market=str(market_id),
            interval=interval,
            startTs=start_ts,
            endTs=end_ts,
        )

    async def stream_market(
        self,
        websocket_url: str,
        asset_ids: list[str],
        on_message: Callable[[dict[str, Any]], Awaitable[None] | None],
        duration_seconds: int | None = None,
    ) -> None:
        if websockets is None:  # pragma: no cover - depende del entorno
            raise RuntimeError("Instala `websockets` para usar el stream de mercado de Polymarket.")
        if not asset_ids:
            return

        async with websockets.connect(websocket_url, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "assets_ids": [str(item) for item in asset_ids],
                        "type": "market",
                    }
                )
            )
            deadline = None if not duration_seconds else (asyncio.get_running_loop().time() + duration_seconds)
            while True:
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    break
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                except TimeoutError:
                    continue
                message = json.loads(raw)
                payloads = message if isinstance(message, list) else [message]
                for payload in payloads:
                    maybe_awaitable = on_message(payload)
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable

    async def stream_sports(
        self,
        websocket_url: str,
        on_message: Callable[[dict[str, Any]], Awaitable[None] | None],
        duration_seconds: int | None = None,
    ) -> None:
        if websockets is None:  # pragma: no cover - depende del entorno
            raise RuntimeError("Instala `websockets` para usar el stream sports de Polymarket.")

        async with websockets.connect(websocket_url, ping_interval=20, ping_timeout=20) as websocket:
            deadline = None if not duration_seconds else (asyncio.get_running_loop().time() + duration_seconds)
            while True:
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    break
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                except TimeoutError:
                    continue
                message = json.loads(raw)
                payloads = message if isinstance(message, list) else [message]
                for payload in payloads:
                    if payload == "PONG":
                        continue
                    maybe_awaitable = on_message(payload)
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
