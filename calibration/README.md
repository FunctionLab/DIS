## Per-disease likelihood ratio (LR) tables for normalized DIS+ predictions 

Wide tables of LR+ per disease × bin:
- [tr](tr/) contains the LR table files for TR-DIS+
- [ptr](ptr/) contains the LR table files for PTR-DIS+

Each folder contains two files:
- **Per-bin LR+** (left): `P(score ∈ bin | pathogenic for this disease) / P(score ∈ bin | neutral)` — same definition as the supplementary figure.
- **Cumulative LR+ at threshold** (right): `P(score > T | pathogenic) / P(score > T | neutral)` 

`NaN` means at least one cell of the 2×2 table was zero for that disease in that bin (almost always: no pathogenic variants for this disease fell in that score range).
