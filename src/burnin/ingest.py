"""CSV/Excel import, column mapping, long<->wide conversion.

Must tolerate the organisers' unknown format: wide `Value_<N>h` columns, arbitrary time points,
single-parameter files, and missing `lot_id` / `part_number` (see build prompt).

Phase 1 fills this in.
"""
