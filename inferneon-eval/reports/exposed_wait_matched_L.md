SIMULATED, descriptive: FLA and reference scored separately, decode only, 20 held-out prompts × 128 tokens per family.
ASSUMED: bandwidth 100e6/413e6 bytes/s; slack 8 ms/layer event; expert payload 1,572,864 bytes; request overhead zero.
Empty cache, empty lane, no prefill/history per prompt; capacities 320/640/1000/2000/3000/4000/6500 slots; one non-preemptive lane.
Metric: summed sequential demand stalls / decode tokens; B1 and perfect IDs use the same L ∈ {0,1,4,40}; oracle is predict(j)=D_j.
Ranking: all three tie at L=0; at every positive-L setting, perfect IDs < B1 = LRU/no hint in exposed wait on both families.
B1 moved nothing: all 112 settings equal no hint in every per-prompt counter; hints repeat still-resident or current-demand IDs.
Capacity matters: 320→6500 slots cuts no-hint wait 79.64% (FLA) / 79.78% (reference), at either assumed bandwidth.
At 1000 slots, L=40: no-hint→perfect wait is 2314.322→1994.489 / 2340.495→2020.613 ms/token at 100 MB/s (FLA/reference).
At 413 MB/s the same cases are 560.369→246.203 / 566.706→251.980 ms/token; perfect prefetch helps, but L=40 is worse than L=1 at 320 slots.
Conclusion: this B1 does not beat the dumb lane; perfect-ID headroom is not a predictor win, and this plot does not select cancel or cap—stop here.
