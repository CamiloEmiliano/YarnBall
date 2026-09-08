# tests/test_app.py
# -*- coding: utf-8 -*-
"""Unit tests for the Chainlit conversational application and PyVis visualizer."""

import pytest
from rag.app import generate_subgraph_html, NODE_COLORS


def test_node_colors_palette():
    """Verify that node color palette contains distinct colors for entity types."""
    assert "Company" in NODE_COLORS
    assert "Product" in NODE_COLORS
    assert "Person" in NODE_COLORS
    assert "Regulation" in NODE_COLORS
    assert NODE_COLORS["Company"].startswith("#")


def test_generate_subgraph_html_with_triples():
    """Test generating a PyVis HTML network canvas from graph triples."""
    triples = [
        {
            "source": "Apple Inc.",
            "relation": "PARTNERED_WITH",
            "target": "TSMC",
            "properties": {"context": "3nm chips"},
        },
        {
            "source": "TSMC",
            "relation": "PRODUCES",
            "target": "A18 Pro",
            "properties": {"date": "2026"},
        },
    ]

    html = generate_subgraph_html(triples)

    assert isinstance(html, str)
    assert len(html) > 100
    assert "Apple Inc." in html
    assert "TSMC" in html
    assert "A18 Pro" in html
    assert "PARTNERED_WITH" in html
    assert "PRODUCES" in html
    assert "#111827" in html  # Background color


def test_generate_subgraph_html_empty():
    """Test generating HTML when no triples are available."""
    html = generate_subgraph_html([])
    assert isinstance(html, str)
    assert "No subgraph connections" in html
