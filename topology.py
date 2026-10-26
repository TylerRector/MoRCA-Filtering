import itertools
import numpy as np
from scipy.linalg import null_space


def rank(a):
    return int(np.linalg.matrix_rank(a, tol=1e-9)) if a.size else 0


def beta1(arcs, n=3):
    arcs = sorted(set(arcs))
    if not arcs:
        return 0
    regular = list(itertools.permutations(range(n), 2))
    forbidden = [x for x in regular if x not in arcs]
    paths = [(i, j, k) for i, j in arcs for j2, k in arcs if j == j2]
    d1 = np.zeros((n, len(arcs)))
    for col, (i, j) in enumerate(arcs):
        d1[i, col], d1[j, col] = -1, 1
    allowed = np.zeros((len(arcs), len(paths)))
    bad = np.zeros((len(forbidden), len(paths)))
    for col, (i, j, k) in enumerate(paths):
        for edge, sign in [((j, k), 1), ((i, k), -1), ((i, j), 1)]:
            if edge[0] == edge[1]:
                continue
            if edge in arcs:
                allowed[arcs.index(edge), col] += sign
            else:
                bad[forbidden.index(edge), col] += sign
    basis = null_space(bad) if forbidden else np.eye(len(paths))
    return len(arcs) - rank(d1) - rank(allowed @ basis)


def aggregate(events):
    weights = {}
    for e in events:
        key = (e['sender'], e['receiver'])
        weights[key] = max(weights.get(key, 0), e['uptake'])
    return weights


def sensitivity(events, index):
    full = aggregate(events)
    deleted = aggregate(events[:index] + events[index + 1:])
    cuts = sorted({0., 1., *full.values(), *deleted.values()})
    result = 0.
    for lo, hi in zip(cuts, cuts[1:]):
        t = (lo + hi) / 2
        a = beta1([e for e, w in full.items() if w >= t])
        b = beta1([e for e, w in deleted.items() if w >= t])
        result += (hi - lo) * min(1, abs(a - b))
    return result


def returns(events, threshold):

    result = []
    def walk(path, seen):
        last = events[path[-1]]
        origin = events[path[0]]['sender']
        if last['receiver'] == origin:
            result.append(path)
            return
        if len(path) == 3:
            return
        for j in range(path[-1] + 1, len(events)):
            e = events[j]
            if e['sender'] != last['receiver'] or e['send_time'] <= last['receive_time']:
                continue
            if e['uptake'] < threshold or (e['receiver'] in seen and e['receiver'] != origin):
                continue
            walk(path + [j], seen | {e['receiver']})
    for i, e in enumerate(events):
        if e['uptake'] >= threshold:
            walk([i], {e['sender'], e['receiver']})
    return result
