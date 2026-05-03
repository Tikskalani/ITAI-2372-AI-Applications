# Data dictionary

The data comes from the NASA Exoplanet Archive's Kepler Object of Interest
(KOI) cumulative table. Each row is one transit-like signal observed by
Kepler, along with whatever the vetting team eventually decided about it.

## The label

`koi_disposition` has three possible values: CONFIRMED, CANDIDATE, and FALSE
POSITIVE. CANDIDATE means the verdict isn't in yet, so I drop those rows. I
map CONFIRMED to 1 and FALSE POSITIVE to 0.

## The ten features I use

| Column | Units | What it is |
|---|---|---|
| koi_period | days | Orbital period of the candidate. |
| koi_duration | hours | How long the transit lasts. |
| koi_depth | parts per million | How much the host star dims during the transit. |
| koi_prad | Earth radii | Inferred planetary radius. |
| koi_teq | Kelvin | Equilibrium temperature of the candidate. |
| koi_insol | Earth flux | Stellar flux the candidate receives. |
| koi_model_snr | dimensionless | Signal-to-noise ratio of the modeled transit. |
| koi_steff | Kelvin | Effective temperature of the host star. |
| koi_slogg | log10(cm/sÂ²) | Surface gravity of the host star. |
| koi_srad | solar radii | Radius of the host star. |

## Why these ten

The full KOI table has 50-something columns and a lot of them are flags or
duplicates of the same physical quantity in different units. I wanted a set
that:

1. Has values for most rows (no point training on a column where 80% of
   entries are NaN).
2. Covers three different aspects of the candidate: the orbit, the planet
   itself, and the host star.
3. Maps onto things astronomers actually use when triaging candidates by
   eye, so the feature importances at the end are interpretable.

## Reference

NASA Exoplanet Archive, KOI cumulative column documentation:
https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
