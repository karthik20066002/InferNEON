"""The matched-L controls; prediction policy stays outside the storage queue.

Only PerfectIDs is the oracle: predict(j) = D_j, the actual demand-time
8-set, not a release-time cache miss set and not future-aware eviction.
Lookahead changes release timing in queue.run_laned, never the target IDs.
"""


class PerfectIDs:
    """Offline demand-ID control. Returning a frozenset prevents mutation."""

    def __init__(self, events):
        self.demands = tuple(frozenset(keys) for _, keys in events)

    def predict(self, j):
        return self.demands[j]


class PreviousToken8:
    """B1: previous decode token's 8 at the SAME layer, not event j-1.

    Start with no history; no prefill or preceding prompt is consulted.
    Valid for matched lookahead <= layers_per_token: the source event is
    known by hint release (at equality, the current event's routed IDs).
    """

    def __init__(self, events, layers_per_token=40):
        if layers_per_token <= 0:
            raise ValueError('layers_per_token must be positive')
        self.demands = tuple(frozenset(keys) for _, keys in events)
        self.layers = layers_per_token

    def predict(self, j):
        return self.demands[j - self.layers] if j >= self.layers else frozenset()


def no_hint(j):
    return frozenset()
