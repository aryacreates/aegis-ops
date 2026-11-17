from collections import deque
from typing import Any

# Directed Dependency Graph: caller_service -> list of services it depends on
SERVICE_DEPENDENCIES: dict[str, list[str]] = {
    "web-storefront": ["checkout-api", "auth-gateway"],
    "mobile-bff": ["checkout-api", "auth-gateway"],
    "partner-webhook": ["checkout-api"],
    "checkout-api": ["redis-cart-cache", "cloud-sql-primary"],
    "auth-gateway": ["cloud-sql-primary"],
    "analytics-etl": ["bigquery-warehouse", "cloud-sql-primary"],
    "redis-cart-cache": [],
    "cloud-sql-primary": [],
    "bigquery-warehouse": [],
}

NODE_CRITICALITY: dict[str, int] = {
    "web-storefront": 10,
    "mobile-bff": 9,
    "checkout-api": 10,
    "auth-gateway": 10,
    "partner-webhook": 6,
    "analytics-etl": 4,
    "redis-cart-cache": 7,
    "cloud-sql-primary": 10,
    "bigquery-warehouse": 5,
}


def compute_blast_radius(target_service: str) -> dict[str, Any]:
    """
    Build reverse dependency adjacency list and execute BFS to determine
    all downstream customer-facing services impacted by an outage in target_service.
    Time Complexity: O(V + E)
    """
    reverse_graph: dict[str, list[str]] = {node: [] for node in SERVICE_DEPENDENCIES}
    for caller, deps in SERVICE_DEPENDENCIES.items():
        for dep in deps:
            reverse_graph.setdefault(dep, []).append(caller)

    if target_service not in reverse_graph:
        return {
            "target_service": target_service,
            "impacted_downstream": [],
            "blast_radius_score": 0,
            "severity": "UNKNOWN",
        }

    visited = {target_service}
    queue = deque([(target_service, 0)])
    impacted: list[dict[str, Any]] = []

    while queue:
        current, hop = queue.popleft()
        for dependent in reverse_graph.get(current, []):
            if dependent not in visited:
                visited.add(dependent)
                impacted.append(
                    {
                        "service": dependent,
                        "hop_distance": hop + 1,
                        "criticality": NODE_CRITICALITY.get(dependent, 5),
                    }
                )
                queue.append((dependent, hop + 1))

    score = NODE_CRITICALITY.get(target_service, 5) + sum(
        item["criticality"] for item in impacted
    )
    severity = (
        "SEV-1 CRITICAL"
        if score >= 25
        else ("SEV-2 HIGH" if score >= 12 else "SEV-3 MODERATE")
    )

    return {
        "target_service": target_service,
        "direct_dependencies": SERVICE_DEPENDENCIES.get(target_service, []),
        "impacted_downstream": impacted,
        "blast_radius_score": score,
        "severity": severity,
    }