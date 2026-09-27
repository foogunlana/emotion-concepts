"""How much noise is in a rate from 32 episodes? Wilson 95% intervals and two-sided Fisher exact tests."""
import math
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * (c - h)), round(100 * (c + h))
def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    return sum(p(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if p(x) <= p(a) * (1 + 1e-9))
for name, k in [("+calm", 27), ("−desperate", 29), ("−disgusted", 17), ("+random", 10)]:
    print(f"{name:11s} {k}/32  95% interval {wilson(k, 32)}")
for name, k1, k2 in [("+calm vs −desperate", 27, 29), ("−disgusted vs +random", 17, 10),
                     ("−lonely vs +random", 20, 10), ("+calm vs unsteered", 27, 2)]:
    print(f"{name:22s} Fisher p = {fisher(k1, 32 - k1, k2, 32 - k2):.2g}")
