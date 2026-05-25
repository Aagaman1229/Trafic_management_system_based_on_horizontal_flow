import json


class TimelineReplay:
    """Provides time-based lookup of cumulative vehicle counts from a
    pre-extracted timeline JSON file.

    The simulation clock drives `sim_time` forward in real time.
    All queries return the cumulative counts at that point in the
    timeline (with wrap-around when the timeline ends).
    """

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.raw = data["timeline"]
        self.roads = sorted(self.raw.keys())
        self.max_time = 0.0
        for entries in self.raw.values():
            if entries:
                self.max_time = max(self.max_time, entries[-1]["t"])
        self._time = 0.0

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if self.max_time > 0:
            self._time = t % self.max_time
        else:
            self._time = 0.0

    def advance(self, dt):
        """Advance the replay clock by *dt* seconds (real time)."""
        self.time = self._time + dt

    def _lookup(self, road, t):
        entries = self.raw.get(road)
        if not entries:
            return {}
        lo, hi = 0, len(entries) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if entries[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid - 1
        return dict(entries[lo]["counts"])

    def get_counts(self, road):
        """Cumulative vehicle counts for *road* at the current replay time."""
        return self._lookup(road, self._time)

    def get_all_counts(self):
        """Returns {road: {vtype: count}} for all roads at current time."""
        return {r: self.get_counts(r) for r in self.roads}

    def get_duration(self):
        return self.max_time
