"""Generator for reference_data/pinch_targets.json.

Cases
1. kemp_four_stream: the four-stream example of Kemp, "Pinch Analysis and Process Integration",
   2nd ed., Butterworth-Heinemann, 2007 (p. 4 of the book per the pina package README; table and
   figure numbers not checked): cold 20 -> 135 C (230 kW), hot 170 -> 60 C (330 kW), cold 80 -> 140 C
   (240 kW), hot 150 -> 30 C (180 kW), dTmin = 10 K, so cp = 2.0, 3.0, 4.0, 1.5 kW/K. Published
   targets typed here: hot utility 20 kW, cold utility 60 kW, pinch 85 C on the shifted scale
   (hot streams 90 C, cold streams 80 C), heating demand 470 kW, cooling demand 510 kW, heat
   recovery 450 kW. These numbers were taken from the pina package README
   (https://github.com/FitzOReilly/pina), which quotes the book, not from the book itself.
2. furman: 25 instances of Furman and Sahinidis (2004), digitised by Letsios, Kouyialis and Misener,
   Computers & Chemical Engineering 113 (2018) 57-85 (data files from their repository, Apache-2.0
   licence, copied to chetools/claude heat-exchanger-network/benchmarks/furman; stream data typed
   below as FURMAN: [t_supply, t_target, cp] with cp in the file's units). Each has one hot and
   one cold utility cost (hu_cost, cu_cost; hu_cost absent = no hot utility) and the utility cost
   published by Letsios et al. (published_cost = hu_cost*QH + cu_cost*QC, their transshipment LP).
   22sp-ph is left out: its published cost differs because the utility temperatures restrict the
   LP, which the problem table does not model.
Independent check: scipy.optimize.linprog (HiGHS) on the heat-cascade LP, written here from scratch:
min cH*R0 + cC*Rn subject to R_k - R_{k-1} = net_k (k = 1..n), R >= 0 (R0 = 0 if no hot utility),
where net_k is the heat released by the process streams in interval k of the shifted scale (cold
streams + dTmin), computed below with numpy independently of the library.
Run: python library/tests/generators/pinch_targets.py
"""
import numpy as np
from scipy.optimize import linprog

from _common import write

FURMAN = {
    '4sp1': {'hot': [[320.0, 200.0, 16.67], [480.0, 280.0, 20.0]], 'cold': [[140.0, 320.0, 14.45], [240.0, 500.0, 11.53]], 'dt_min': 10.0, 'hu_cost': 0.001, 'cu_cost': 5e-05, 'published_cost': 0.383275},
    '6sp-cf1': {'hot': [[500.0, 350.0, 10.0], [450.0, 350.0, 12.0], [400.0, 320.0, 8.0]], 'cold': [[300.0, 480.0, 9.0], [340.0, 420.0, 10.0], [340.0, 400.0, 8.0]], 'dt_min': 10.0, 'hu_cost': 80.0, 'cu_cost': 20.0, 'published_cost': 8800.0},
    '6sp-gg1': {'hot': [[300.0, 200.0, 10.0], [200.0, 190.0, 100.0], [190.0, 170.0, 50.0]], 'cold': [[160.0, 180.0, 50.0], [180.0, 190.0, 100.0], [190.0, 230.0, 25.0]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 0.0},
    '6sp1': {'hot': [[440.0, 150.0, 28.0], [520.0, 300.0, 23.8], [390.0, 150.0, 33.6]], 'cold': [[100.0, 430.0, 16.0], [180.0, 330.0, 32.76], [200.0, 400.0, 26.35]], 'dt_min': 10.0, 'hu_cost': 0.003, 'cu_cost': 5e-05, 'published_cost': 0.2978},
    '7sp-cm1': {'hot': [[626.0, 586.0, 9.802], [620.0, 519.0, 2.931], [528.0, 353.0, 6.161]], 'cold': [[497.0, 613.0, 7.179], [389.0, 576.0, 0.641], [326.0, 386.0, 7.627], [313.0, 566.0, 1.69]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 293.507},
    '7sp-s1': {'hot': [[101.0, 40.0, 95.6], [193.0, 30.0, 64.8], [158.0, 30.0, 62.6], [346.0, 30.0, 56.1], [263.0, 50.0, 115.4], [419.0, 80.0, 174.8]], 'cold': [[30.0, 390.0, 573.0]], 'dt_min': 10.0, 'hu_cost': 5.0, 'cu_cost': 2.5, 'published_cost': 415303.5},
    '7sp-torw1': {'hot': [[160.0, 110.0, 7.032], [249.0, 138.0, 8.44], [227.0, 106.0, 11.816], [271.0, 146.0, 5.6]], 'cold': [[96.0, 160.0, 9.144], [116.0, 217.0, 7.296], [140.0, 250.0, 18.0]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 578.784},
    '7sp1': {'hot': [[440.0, 150.0, 28.0], [520.0, 300.0, 23.8], [390.0, 150.0, 33.6]], 'cold': [[100.0, 430.0, 16.0], [180.0, 350.0, 32.76], [200.0, 400.0, 26.35], [350.0, 410.0, 19.84]], 'dt_min': 10.0, 'cu_cost': 5e-05, 'published_cost': 0.20552},
    '7sp2': {'hot': [[590.0, 400.0, 23.76], [471.0, 200.0, 15.77], [533.0, 150.0, 13.2]], 'cold': [[200.0, 400.0, 16.0], [100.0, 430.0, 16.0], [300.0, 400.0, 41.28], [150.0, 280.0, 26.24]], 'dt_min': 10.0, 'hu_cost': 0.001, 'cu_cost': 5e-05, 'published_cost': 2.17553},
    '7sp4': {'hot': [[630.555, 338.888, 7.913], [583.333, 505.555, 5.803], [555.555, 319.444, 2.374], [494.444, 447.222, 31.652], [477.777, 311.111, 6.3305], [422.222, 383.333, 65.943]], 'cold': [[288.888, 650.0, 24.795]], 'dt_min': 10.0, 'hu_cost': 2341.84, 'cu_cost': 1822.36, 'published_cost': 9178080.285},
    '8sp-fs1': {'hot': [[109.0, 40.0, 42.03], [295.0, 45.0, 3.24], [271.0, 60.0, 10.89], [287.0, 60.0, 1.98], [361.0, 190.0, 31.58]], 'cold': [[66.0, 340.0, 40.51], [10.0, 95.0, 4.7], [137.0, 138.0, 1000.0]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 4645.2},
    '8sp1': {'hot': [[470.0, 320.0, 22.4], [450.0, 240.0, 17.5], [370.0, 150.0, 28.5], [310.0, 200.0, 20.1]], 'cold': [[200.0, 420.0, 16.8], [150.0, 400.0, 23.2], [185.0, 330.0, 35.1], [140.0, 300.0, 17.25]], 'dt_min': 10.0, 'hu_cost': 0.02, 'cu_cost': 5e-05, 'published_cost': 38.845625},
    '9sp-al1': {'hot': [[327.0, 40.0, 0.1], [220.0, 160.0, 0.16], [220.0, 60.0, 0.06], [160.0, 60.0, 0.4]], 'cold': [[100.0, 300.0, 0.1], [35.0, 164.0, 0.07], [85.0, 138.0, 0.35], [60.0, 170.0, 0.06], [140.0, 300.0, 0.2]], 'dt_min': 10.0, 'hu_cost': 50000.0, 'cu_cost': 7000.0, 'published_cost': 997000.0},
    '9sp-has1': {'hot': [[120.0, 65.0, 50.0], [80.0, 50.0, 300.0], [135.0, 110.0, 290.0], [220.0, 95.0, 20.0], [135.0, 105.0, 260.0]], 'cold': [[65.0, 90.0, 150.0], [75.0, 200.0, 140.0], [30.0, 210.0, 100.0], [60.0, 140.0, 50.0]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 22950.0},
    '10sp-la1': {'hot': [[327.0, 40.0, 0.1], [220.0, 160.0, 0.16], [220.0, 60.0, 0.06], [160.0, 60.0, 0.4]], 'cold': [[100.0, 300.0, 0.1], [35.0, 164.0, 0.07], [85.0, 138.0, 0.35], [60.0, 170.0, 0.06], [140.0, 300.0, 0.2]], 'dt_min': 10.0, 'hu_cost': 75000.0, 'cu_cost': 10000.0, 'published_cost': 1486000.0},
    '10sp-ol1': {'hot': [[327.0, 30.0, 0.1], [220.0, 160.0, 0.25], [220.0, 60.0, 0.02], [160.0, 45.0, 0.34]], 'cold': [[100.0, 300.0, 0.2], [35.0, 164.0, 0.07], [80.0, 125.0, 0.175], [60.0, 170.0, 0.06], [140.0, 300.0, 0.2], [10.0, 50.0, 0.3]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 39.455},
    '10sp1': {'hot': [[320.0, 200.0, 16670.0], [480.0, 280.0, 20000.0], [440.0, 150.0, 28000.0], [520.0, 300.0, 23800.0], [390.0, 150.0, 33600.0]], 'cold': [[140.0, 320.0, 14450.0], [240.0, 431.0, 11530.0], [100.0, 430.0, 16000.0], [180.0, 350.0, 32760.0], [200.0, 400.0, 26350.0]], 'dt_min': 10.0, 'cu_cost': 5e-05, 'published_cost': 324.8985},
    '12sp1': {'hot': [[668.0, 200.0, 402.678], [596.0, 258.0, 222.318], [434.0, 240.0, 341.26], [301.0, 220.0, 553.66], [490.0, 150.0, 37.725], [460.0, 415.0, 90.645], [405.0, 150.0, 79.546], [323.0, 290.0, 92.034], [290.0, 150.0, 85.303]], 'cold': [[40.0, 240.0, 620.198], [240.0, 695.0, 822.659], [108.0, 354.0, 138.299]], 'dt_min': 10.0, 'hu_cost': 0.02, 'cu_cost': 1e-05, 'published_cost': 2111.08028},
    '14sp1': {'hot': [[480.0, 300.0, 18.4], [460.0, 200.0, 24.18], [450.0, 300.0, 20.8], [400.0, 190.0, 18.2], [390.0, 250.0, 28.22], [320.0, 125.0, 19.17], [280.0, 125.0, 27.2]], 'cold': [[150.0, 430.0, 13.3], [180.0, 400.0, 13.02], [220.0, 400.0, 27.2], [110.0, 350.0, 22.2], [140.0, 390.0, 28.16], [130.0, 280.0, 15.6], [100.0, 240.0, 13.05]], 'dt_min': 10.0, 'cu_cost': 0.0005, 'published_cost': 0.213175},
    '15sp-tkm': {'hot': [[137.0, 14.0, 5.3], [135.0, 66.0, 16.8], [111.0, 110.0, 3327.0], [105.0, 95.0, 135.4], [76.0, 66.0, 134.6], [75.0, 74.0, 928.0], [66.0, 65.0, 840.0], [65.0, 64.0, 12000.0], [37.0, 14.0, 324.0]], 'cold': [[4.0, 112.0, 56.6], [76.0, 100.0, 108.1], [70.0, 90.0, 205.0], [38.0, 85.0, 18.4], [45.0, 70.0, 280.1], [4.0, 45.0, 314.0]], 'dt_min': 10.0, 'hu_cost': 88.0, 'cu_cost': 1.0, 'published_cost': 514246.1},
    '20sp1': {'hot': [[550.0, 450.0, 28.4], [520.0, 480.0, 23.87], [500.0, 440.0, 24.6], [460.0, 370.0, 17.0], [415.0, 340.0, 30.69], [400.0, 300.0, 19.36], [365.0, 320.0, 25.5], [300.0, 250.0, 12.42], [240.0, 170.0, 20.72], [170.0, 140.0, 18.9]], 'cold': [[375.0, 420.0, 22.32], [300.0, 370.0, 23.87], [320.0, 360.0, 34.96], [280.0, 340.0, 26.04], [260.0, 330.0, 13.44], [225.0, 280.0, 32.0], [240.0, 265.0, 11.1], [170.0, 220.0, 22.96], [140.0, 200.0, 14.4], [80.0, 140.0, 13.92]], 'dt_min': 10.0, 'cu_cost': 0.0005, 'published_cost': 1.681425},
    '22sp1': {'hot': [[137.8, 51.7, 5.28], [160.0, 51.7, 10.12], [232.2, 160.0, 12.93], [187.8, 87.8, 9.6], [232.2, 148.9, 10.98], [160.0, 93.3, 21.1], [248.9, 187.8, 7.92], [154.4, 93.3, 10.47], [160.0, 65.6, 15.04], [232.2, 115.6, 13.19], [243.3, 160.0, 4.414]], 'cold': [[37.8, 115.6, 6.89], [54.4, 137.8, 8.23], [65.6, 148.9, 14.86], [43.3, 173.9, 9.23], [173.9, 215.6, 18.47], [82.2, 204.4, 6.87], [173.9, 260.0, 22.56], [60.0, 148.9, 9.1], [85.0, 173.9, 6.33], [65.6, 204.4, 12.23], [176.7, 260.0, 19.81]], 'dt_min': 10.0, 'hu_cost': 0.0022, 'cu_cost': 0.00011, 'published_cost': 5.284960846},
    '23sp1': {'hot': [[433.0, 366.0, 8.79], [522.0, 411.0, 10.55], [510.0, 349.0, 14.77], [544.0, 422.0, 12.56], [472.0, 339.0, 17.73], [505.0, 339.0, 14.77], [544.0, 420.0, 12.56], [475.0, 339.0, 17.73], [583.0, 478.0, 12.53], [517.0, 366.0, 8.32], [511.0, 339.0, 6.96]], 'cold': [[323.0, 423.0, 7.62], [389.0, 495.0, 6.08], [311.0, 494.0, 8.44], [355.0, 450.0, 17.28], [366.0, 478.0, 13.9], [311.0, 490.0, 8.44], [350.0, 450.0, 17.28], [352.0, 468.0, 13.9], [366.0, 478.0, 8.44], [311.0, 489.0, 8.44], [422.0, 478.0, 21.78], [339.0, 411.0, 13.84]], 'dt_min': 10.0, 'cu_cost': 1.0, 'published_cost': 2553.67},
    '28sp-as1': {'hot': [[230.0, 80.0, 30.0], [200.0, 40.0, 45.0], [110.0, 45.0, 0.1], [115.0, 40.0, 0.1], [105.0, 40.0, 0.1], [110.0, 42.0, 0.1], [117.0, 48.0, 0.1], [103.0, 50.0, 0.12], [110.0, 45.0, 0.1], [115.0, 40.0, 0.1], [105.0, 40.0, 0.1], [110.0, 42.0, 0.1], [117.0, 48.0, 0.1], [103.0, 50.0, 0.1], [115.0, 42.0, 0.1], [117.0, 43.0, 0.1]], 'cold': [[40.0, 180.0, 40.0], [140.0, 280.0, 60.0], [170.0, 270.0, 0.1], [175.0, 265.0, 0.1], [180.0, 275.0, 0.1], [168.0, 277.0, 0.1], [181.0, 267.0, 0.1], [170.0, 270.0, 0.1], [175.0, 265.0, 0.1], [180.0, 275.0, 0.1], [168.0, 277.0, 0.1], [181.0, 267.0, 0.1]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 8590.76},
    '37sp-yfyv': {'hot': [[175.0, 150.0, 4923.0], [168.0, 55.0, 767.0], [190.0, 175.0, 15796.0], [160.0, 135.0, 6402.0], [102.0, 40.0, 40417.0], [170.0, 40.0, 795.0], [98.4, 40.0, 2216.0], [97.4, 40.0, 2290.0], [98.4, 40.0, 1984.0], [40.0, 25.0, 652.0], [25.0, 10.0, 692.0], [104.7, 40.0, 573.0], [80.0, 40.0, 840.0], [40.0, 25.0, 339.0], [25.0, 7.4, 1553.0], [57.5, 40.0, 358.0], [11.7, 0.0, 1819.0], [99.0, 96.2, 529.0], [30.8, 28.6, 6690.0], [44.7, 40.0, 5307.0], [976.0, 300.0, 25220.0]], 'cold': [[10.0, 160.0, 4923.0], [170.0, 170.5, 15796.0], [120.0, 120.5, 6402.0], [170.0, 170.5, 13667.0], [170.0, 222.0, 1604.0], [68.0, 90.0, 778.0], [14.5, 15.0, 1448.0], [122.0, 122.5, 386.0], [2.4, 2.6, 1625.0], [68.0, 68.5, 3241.0], [4.5, 5.0, 3262.0], [52.0, 53.0, 4666.0], [175.0, 196.0, 6300.0], [40.0, 99.0, 6490.0], [33.9, 79.9, 1875.0], [130.0, 302.0, 12703.0]], 'dt_min': 10.0, 'hu_cost': 1.0, 'cu_cost': 1.0, 'published_cost': 17180884.3},
}

def cascade_lp(hot, cold, dt, c_h, c_c):
    """Heat-cascade LP; returns (QH, QC) from HiGHS."""
    hot, cold = np.array(hot, float).reshape(-1, 3), np.array(cold, float).reshape(-1, 3)
    edges = np.unique(np.concatenate([hot[:, :2].ravel(), cold[:, :2].ravel() + dt]))[::-1]
    n = len(edges) - 1
    net = np.zeros(n)
    for k in range(n):
        hi, lo = edges[k], edges[k + 1]
        for ts, tt, cp in hot:
            net[k] += cp * max(0.0, min(hi, ts) - max(lo, tt))
        for ts, tt, cp in cold:
            net[k] -= cp * max(0.0, min(hi, tt + dt) - max(lo, ts + dt))
    # variables R_0..R_n; equalities R_k - R_{k-1} = net_k
    A = np.zeros((n, n + 1))
    for k in range(n):
        A[k, k + 1], A[k, k] = 1.0, -1.0
    cost = np.zeros(n + 1)
    cost[0], cost[n] = c_h if c_h is not None else 1.0, c_c
    bounds = [(0, None)] * (n + 1)
    if c_h is None:
        bounds[0] = (0, 0)
    res = linprog(cost, A_eq=A, b_eq=net, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return float(res.x[0]), float(res.x[n])


kemp = {"hot": [[170.0, 60.0, 3.0], [150.0, 30.0, 1.5]], "cold": [[20.0, 135.0, 2.0], [80.0, 140.0, 4.0]],
        "dt_min": 10.0,
        "published": {"qh_min": 20.0, "qc_min": 60.0, "pinch_shifted": 85.0, "pinch_hot": 90.0,
                      "heating_demand": 470.0, "cooling_demand": 510.0, "heat_recovery": 450.0}}
qh, qc = cascade_lp(kemp["hot"], kemp["cold"], 10.0, 1.0, 0.01)
kemp.update(lp_qh=qh, lp_qc=qc)
assert abs(qh - 20) < 1e-7 and abs(qc - 60) < 1e-7

furman = []
for name, d in FURMAN.items():
    qh, qc = cascade_lp(d["hot"], d["cold"], d["dt_min"], d.get("hu_cost"), d["cu_cost"])
    cost = qh * d.get("hu_cost", 0.0) + qc * d["cu_cost"]
    assert abs(cost - d["published_cost"]) <= 1e-9 * abs(d["published_cost"]), name
    furman.append(dict(name=name, **d, lp_qh=qh, lp_qc=qc, lp_cost=cost))

write("pinch_targets", "library/tests/generators/pinch_targets.py",
      {"lp": "scipy.optimize.linprog method=highs", "furman_instances": len(furman),
       "furman_source": "Letsios, Kouyialis, Misener (2018), Comput. Chem. Eng. 113, 57-85; Apache-2.0",
       "kemp_source": "Kemp (2007), Pinch Analysis and Process Integration, 2nd ed., four-stream example"},
      "Streams [t_supply, t_target, cp]: temperatures in C or K (as in the source), cp in kW/K (Furman "
      "instances: the file's own units); dt_min as temperature difference; qh/qc in the same energy-rate unit.",
      {"kemp_four_stream": kemp, "furman": furman})
