# Dataset Manifest

Large generated arrays are not committed to Git. The experiment server retained the following datasets.

| Setting | Server directory | Principal array shape | dtype |
|---|---|---:|---|
| ET SIS | `/root/autodl-tmp/qkzhang/SIS_data_0222/` | `(2500, 98304)` per lensed image | float64 |
| ET PM | `/root/autodl-tmp/qkzhang/PM_data_0222/` | `(2500, 98304)` per lensed image | float64 |
| ET unlensed | `/root/autodl-tmp/qkzhang/Unlensed_data_0222/` | `(5000, 98304)` | float64 |
| LIGO SIS | `ET_vs_LIGO/LIGO_data/SIS_data/` | `(10000, 2, 98304)` per lensed image | float64 |
| LIGO PM | `ET_vs_LIGO/LIGO_data/PM_data/` | `(10000, 2, 98304)` per lensed image | float64 |
| LIGO unlensed | `ET_vs_LIGO/LIGO_data/Unlensed_data/` | `(10000, 2, 98304)` | float64 |

Each LIGO sample contains H1 and L1 detector channels. The final LIGO generation scripts sample source-frame component masses from 10 to 100 solar masses; Table 1 of the current paper draft states 10 to 70 solar masses for the shared population description.

The GitHub release is therefore an artifact-and-code archive, not a self-contained dataset release. A durable external archive should preserve the arrays with checksums and a DOI before publication.
