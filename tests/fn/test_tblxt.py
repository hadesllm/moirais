"""Tests for moirais.fn.tblxt -- HTML table extractor."""

from moirais.fn.tblxt import extract_tables, tblxt


class TestTblxt:
    def test_basic_table(self):
        html = """
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Alice</td><td>30</td></tr>
            <tr><td>Bob</td><td>25</td></tr>
            <tr><td>Carol</td><td>35</td></tr>
        </table>
        """
        tables = extract_tables(html)
        assert len(tables) == 1
        df = tables[0]
        assert df.shape == (3, 2)
        assert list(df.columns) == ["Name", "Age"]

    def test_no_tables(self):
        tables = extract_tables("<html><body>No tables</body></html>")
        assert tables == []

    def test_result_wrapper(self):
        html = "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        result = tblxt(html)
        assert result.value == 1
