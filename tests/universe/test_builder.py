"""
Unit tests for universe/builder.py constituent dataset builder.
"""

from pathlib import Path
import tempfile
import pytest

from universe.builder import extract_wiki_constituents, get_historical_turnover_records


SAMPLE_WIKI_HTML = """
<html>
<body>
<table id="constituents">
  <tr>
    <th>Symbol</th>
    <th>Security</th>
    <th>GICS Sector</th>
    <th>GICS Sub-Industry</th>
    <th>Headquarters Location</th>
    <th>Date added</th>
    <th>CIK</th>
    <th>Founded</th>
  </tr>
  <tr>
    <td>AAPL</td>
    <td>Apple Inc.</td>
    <td>Information Technology</td>
    <td>Technology Hardware, Storage & Peripherals</td>
    <td>Cupertino, California</td>
    <td>1982-11-30</td>
    <td>0000320193</td>
    <td>1976</td>
  </tr>
  <tr>
    <td>BRK.B</td>
    <td>Berkshire Hathaway</td>
    <td>Financials</td>
    <td>Multi-Sector Holdings</td>
    <td>Omaha, Nebraska</td>
    <td>2010-02-16</td>
    <td>0001067983</td>
    <td>1839</td>
  </tr>
</table>
</body>
</html>
"""


def test_get_historical_turnover_records():
    records = get_historical_turnover_records()
    assert len(records) > 10
    tickers = {r["ticker"] for r in records}
    assert "TWTR" in tickers
    assert "SIVB" in tickers
    assert "ATVI" in tickers
    for r in records:
        assert r["is_current"] is False
        assert r["date_removed"] is not None


def test_extract_wiki_constituents(tmp_path: Path):
    html_file = tmp_path / "wiki_test.html"
    html_file.write_text(SAMPLE_WIKI_HTML, encoding="utf-8")

    constituents = extract_wiki_constituents(html_file)
    assert len(constituents) == 2

    aapl = next(c for c in constituents if c["ticker"] == "AAPL")
    assert aapl["cik"] == "0000320193"
    assert aapl["company_name"] == "Apple Inc."
    assert aapl["gics_sector"] == "Information Technology"
    assert aapl["is_current"] is True

    brk = next(c for c in constituents if c["ticker"] == "BRK-B")
    assert brk["cik"] == "0001067983"
    assert brk["gics_sector"] == "Financials"


def test_extract_wiki_constituents_missing_table(tmp_path: Path):
    empty_file = tmp_path / "empty.html"
    empty_file.write_text("<html><body><p>No table</p></body></html>", encoding="utf-8")

    with pytest.raises(ValueError, match="Table 'constituents' not found"):
        extract_wiki_constituents(empty_file)
