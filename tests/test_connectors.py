import httpx
import pandas as pd
import pytest
from sqlalchemy import create_engine

from hydromcp.connectors import files
from hydromcp.connectors.pi import PIClient
from hydromcp.connectors.sql import query_sql


def test_csv_autodetect_and_bad_values(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("Date,TI1,AI2\n2025-01-01 00:00,650,9.1\n2025-01-01 01:00,651,Bad Input\n"
                 "2025-01-01 02:00,652,8.7\n")
    df = files.load_file(str(p))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df["AI2"].isna().sum() == 1 and df["AI2"].dtype.kind == "f"


def test_excel(tmp_path):
    p = tmp_path / "d.xlsx"
    pd.DataFrame({"time": pd.date_range("2025-01-01", periods=3, freq="h"),
                  "x": [1, 2, 3]}).to_excel(p, index=False, sheet_name="data")
    assert files.list_sheets(str(p)) == ["data"]
    assert files.load_file(str(p), time_column="time", sheet="data")["x"].sum() == 6


def test_sql_readonly(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'lims.db'}"
    eng = create_engine(url)
    pd.DataFrame({"ts": ["2025-01-01", "2025-01-02"], "sulfur": [9.0, 8.0]}).to_sql("lab", eng)
    eng.dispose()
    monkeypatch.setenv("HYDROMCP_SQL_LIMS", url)
    df = query_sql("lims", "SELECT ts, sulfur FROM lab", time_column="ts")
    assert list(df["sulfur"]) == [9.0, 8.0]
    for bad in ("DELETE FROM lab", "SELECT 1; DROP TABLE lab", "drop table lab"):
        with pytest.raises(ValueError):
            query_sql("lims", bad)
    with pytest.raises(ValueError, match="HYDROMCP_SQL_NOPE"):
        query_sql("nope", "SELECT 1")


def test_pi_client_mocked():
    def handler(req: httpx.Request):
        if req.url.path.endswith("/points"):
            return httpx.Response(200, json={"WebId": "W1"})
        if "/streams/W1/interpolated" in req.url.path:
            return httpx.Response(200, json={"Items": [
                {"Timestamp": "2025-01-01T00:00:00Z", "Value": 650.1},
                {"Timestamp": "2025-01-01T01:00:00Z", "Value": {"Name": "Bad Input", "Value": 307}},
                {"Timestamp": "2025-01-01T02:00:00Z", "Value": "I/O Timeout"},
            ]})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pi = PIClient(base_url="https://pi/piwebapi", client=client, data_server="PISRV")
    df = pi.fetch(["11TI110.PV"], "*-1d", "*", "1h")
    assert df.shape == (3, 1)
    assert df.iloc[0, 0] == 650.1
    # digital state dicts return their numeric code; strings become NaN
    assert df.iloc[1, 0] == 307 and pd.isna(df.iloc[2, 0])
