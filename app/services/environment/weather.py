"""Environment Weather Service and Risk Vector for Tree of Life."""
import json
import time
from pathlib import Path
from typing import Optional, Dict
import httpx
from app.schemas.environment import EnvironmentResult

# Karnataka district coordinates lookup
DISTRICT_COORDS = {
    "mandya": {"lat": 12.52, "lon": 76.89},
    "mysuru": {"lat": 12.29, "lon": 76.63},
    "shivamogga": {"lat": 13.92, "lon": 75.56},
    "shimoga": {"lat": 13.92, "lon": 75.56},
    "raichur": {"lat": 16.20, "lon": 77.35},
    "ballari": {"lat": 15.13, "lon": 76.92},
    "bellary": {"lat": 15.13, "lon": 76.92},
    "davangere": {"lat": 14.46, "lon": 75.92},
    "hassan": {"lat": 13.00, "lon": 76.10},
    "koppal": {"lat": 15.34, "lon": 76.15},
    "udupi": {"lat": 13.34, "lon": 74.74},
}


class EnvironmentService:
    """Fetches microclimate conditions and generates disease risk vectors."""
    def __init__(self, cache_dir: str = "data/cache/weather"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get_district_risk(self, district: Optional[str]) -> Optional[EnvironmentResult]:
        if not district:
            return None

        d_key = district.lower().strip()
        coords = DISTRICT_COORDS.get(d_key, {"lat": 12.97, "lon": 77.59})  # Default to Karnataka center

        cache_file = self.cache_dir / f"{d_key}.json"
        now = time.time()

        # Check 24-hour disk cache
        weather_data = None
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if now - cached.get("cached_at", 0) < 86400:  # 24 hours
                        weather_data = cached["data"]
            except Exception:
                weather_data = None

        if not weather_data:
            try:
                url = (
                    f"https://api.open-meteo.com/v1/forecast?"
                    f"latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,precipitation"
                )
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        raw = resp.json().get("current", {})
                        weather_data = {
                            "temp": raw.get("temperature_2m", 28.0),
                            "humidity": raw.get("relative_humidity_2m", 75.0),
                            "precip": raw.get("precipitation", 2.0)
                        }
                        # Save to cache
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump({"cached_at": now, "data": weather_data}, f)
            except Exception:
                # Fallback to typical Karnataka rice climate
                weather_data = {"temp": 28.5, "humidity": 78.0, "precip": 1.5}

        temp = weather_data.get("temp", 28.0)
        rh = weather_data.get("humidity", 75.0)

        # Rule-based disease risk heuristic
        # Warm (25-30C) + high humidity (>80%) creates high blast & BLB pressure
        blast_risk = 0.50
        blb_risk = 0.50
        brown_spot_risk = 0.40
        scald_risk = 0.35
        tungro_risk = 0.30

        if rh >= 80.0 and 24.0 <= temp <= 30.0:
            blast_risk = 0.80
            blb_risk = 0.75
        elif rh >= 70.0:
            blast_risk = 0.65
            blb_risk = 0.60

        risk_vector = {
            "rice_blast": blast_risk,
            "rice_brown_spot": brown_spot_risk,
            "bacterial_leaf_blight": blb_risk,
            "rice_leaf_scald": scald_risk,
            "rice_tungro": tungro_risk,
        }

        return EnvironmentResult(
            used=True,
            district=district,
            temperature_celsius=temp,
            relative_humidity_pct=rh,
            precipitation_mm=weather_data.get("precip", 0.0),
            risk_vector=risk_vector,
            source="open_meteo_cached",
            is_rule_based_provisional=True
        )
