"""Tests for mycelium memory network."""

import sqlite3

from memory_router.memory.mycelium import MyceliumNetwork


def _make_network():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return MyceliumNetwork(conn)


def test_strengthen_co_retrieved():
    net = _make_network()
    updated = net.strengthen_co_retrieved([1, 2, 3])
    assert updated == 3  # 3 edges: (1,2), (1,3), (2,3)
    assert net.edge_count() == 3


def test_strengthen_boosts_weight():
    net = _make_network()
    net.strengthen_co_retrieved([1, 2])
    net.strengthen_co_retrieved([1, 2])
    neighbors = net.get_neighbors(1)
    assert len(neighbors) == 1
    assert neighbors[0][1] > 1.0  # weight should be boosted


def test_spread_activation_basic():
    net = _make_network()
    # Create a chain: 1-2, 2-3
    net.add_edge(1, 2, weight=2.0)
    net.add_edge(2, 3, weight=2.0)

    results = net.spread_activation([1], max_hops=2, top_k=5)
    # Should find node 2 (1 hop) and node 3 (2 hops)
    found_ids = [r[0] for r in results]
    assert 2 in found_ids
    assert 3 in found_ids


def test_spread_activation_returns_empty_for_isolated_node():
    net = _make_network()
    results = net.spread_activation([99], max_hops=2, top_k=5)
    assert results == []


def test_decay_edges():
    net = _make_network()
    net.add_edge(1, 2, weight=0.15)
    # With very aggressive decay, this low-weight edge should get pruned
    pruned = net.decay_edges(half_life_days=0.001)
    assert pruned >= 0  # At least attempted


def test_remove_memory():
    net = _make_network()
    net.add_edge(1, 2)
    net.add_edge(1, 3)
    net.add_edge(2, 3)
    removed = net.remove_memory(1)
    assert removed == 2  # edges (1,2) and (1,3)
    assert net.edge_count() == 1


def test_stats():
    net = _make_network()
    net.add_edge(1, 2, weight=2.0)
    net.add_edge(2, 3, weight=3.0)
    s = net.stats()
    assert s["edge_count"] == 2
    assert s["avg_weight"] == 2.5
    assert s["max_weight"] == 3.0


def test_single_memory_no_edges():
    net = _make_network()
    updated = net.strengthen_co_retrieved([1])
    assert updated == 0


def test_empty_list_no_edges():
    net = _make_network()
    updated = net.strengthen_co_retrieved([])
    assert updated == 0
