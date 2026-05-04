# QuickDraw Results

Generated from local QuickDraw `summary.json` artifacts under `AGE_Arc-based_Geo-entity_Encoder-Alex/model_edges/results/` and collected into one repository-side markdown summary.

- Total QuickDraw result files: **102**
- Experiment groups: **20**
- Source snapshot root: `D:\ArcSet\AGE_Arc-based_Geo-entity_Encoder-Alex\model_edges\results`

## Group Coverage

| Group | Files | Best Test Acc | Best Run |
|---|---:|---:|---|
| root_quickdraw_runs | 2 | 0.7795 | deepset_single_quickdraw_e50 |
| quickdraw_100c100s | 2 | 0.4093 | deepset_single_quickdraw_e80 |
| quickdraw_deepset_lowshot | 5 | 0.6887 | 100c1000s/deepset_single_quickdraw_e80 |
| quickdraw_isab_lowshot | 3 | 0.4073 | 100c200s/settransformer_isab_single_quickdraw_e80 |
| quickdraw_sab_lowshot | 3 | 0.4750 | 100c200s/settransformer_sab_single_quickdraw_e80 |
| quickdraw_pointnet_lowshot | 10 | 0.7175 | 100c1000s/pointnet_single_quickdraw_e80 |
| quickdraw_settransformer_stability | 13 | 0.6841 | settransformer-sab/final/lr5e-4_warm5_clip1/100c1000s/settransformer_sab_single_quickdraw_e80 |
| quickdraw_settransformer_multiseed | 9 | 0.4080 | settransformer-sab/old_default/100c200s/seed1569/settransformer_sab_single_quickdraw_e40 |
| quickdraw_settransformer_quality_probe | 12 | 0.4637 | settransformer-sab/mean_pool_ls010/100c200s/settransformer_sab_single_quickdraw_e30 |
| quickdraw_settransformer_scheduler_probe | 6 | 0.6127 | settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c500s/settransformer_sab_single_quickdraw_e80 |
| quickdraw_settransformer_lowshot_stable | 10 | 0.7058 | 100c1000s/settransformer_isab_single_quickdraw_e80 |
| quickdraw_settransformer_lowshot_stable_meanpool | 10 | 0.6990 | 100c1000s/settransformer_isab_single_quickdraw_e80 |
| quickdraw_stability_probe | 1 | 0.5403 | sab_pma_lr3e4_warm5_clip1_100c500s/settransformer_sab_single_quickdraw_e12 |
| smoke_quickdraw_sweep | 3 | 0.0067 | deepset_single_quickdraw_e1 |
| smoke_quickdraw_sweep_b512 | 3 | 0.0100 | deepset_single_quickdraw_e1 |
| smoke_quickdraw_sweep_b1024 | 2 | 0.0113 | settransformer_sab_single_quickdraw_e1 |
| smoke_quickdraw_sweep_padded_b512 | 3 | 0.0100 | deepset_single_quickdraw_e1 |
| smoke_quickdraw_pointnet_lowshot | 2 | 0.0173 | 100c50s/pointnet2_single_quickdraw_e1 |
| smoke_quickdraw_settransformer_lowshot_stable | 2 | 0.0093 | 100c50s/settransformer_sab_single_quickdraw_e1 |
| smoke_quickdraw_settransformer_lowshot_stable_meanpool | 1 | 0.0093 | 100c50s/settransformer_sab_single_quickdraw_e1 |

## Top Runs Overall

| Rank | Group | Variant | Model | Subset | Test Acc | Macro F1 | Best Epoch |
|---:|---|---|---|---|---:|---:|---:|
| 1 | root_quickdraw_runs | deepset_single_quickdraw_e50 | deepset | 100c10000s | 0.7795 | 0.7790 | 48 |
| 2 | quickdraw_pointnet_lowshot | 100c1000s/pointnet_single_quickdraw_e80 | pointnet | 100c1000s | 0.7175 | 0.7186 | 68 |
| 3 | quickdraw_pointnet_lowshot | 100c1000s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c1000s | 0.7073 | 0.7066 | 59 |
| 4 | quickdraw_settransformer_lowshot_stable | 100c1000s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c1000s | 0.7058 | 0.7040 | 24 |
| 5 | quickdraw_settransformer_lowshot_stable_meanpool | 100c1000s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c1000s | 0.6990 | 0.6973 | 27 |
| 6 | quickdraw_settransformer_lowshot_stable_meanpool | 100c1000s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c1000s | 0.6962 | 0.6950 | 25 |
| 7 | quickdraw_deepset_lowshot | 100c1000s/deepset_single_quickdraw_e80 | deepset | 100c1000s | 0.6887 | 0.6875 | 46 |
| 8 | quickdraw_settransformer_lowshot_stable | 100c1000s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c1000s | 0.6867 | 0.6859 | 26 |
| 9 | quickdraw_settransformer_stability | settransformer-sab/final/lr5e-4_warm5_clip1/100c1000s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c1000s | 0.6841 | 0.6816 | 20 |
| 10 | quickdraw_pointnet_lowshot | 100c500s/pointnet_single_quickdraw_e80 | pointnet | 100c500s | 0.6773 | 0.6770 | 63 |
| 11 | quickdraw_pointnet_lowshot | 100c500s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c500s | 0.6588 | 0.6562 | 78 |
| 12 | quickdraw_settransformer_lowshot_stable | 100c500s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c500s | 0.6357 | 0.6343 | 29 |
| 13 | quickdraw_settransformer_lowshot_stable_meanpool | 100c500s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c500s | 0.6325 | 0.6307 | 30 |
| 14 | quickdraw_deepset_lowshot | 100c500s/deepset_single_quickdraw_e80 | deepset | 100c500s | 0.6223 | 0.6209 | 22 |
| 15 | quickdraw_settransformer_lowshot_stable_meanpool | 100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 0.6191 | 0.6164 | 23 |
| 16 | quickdraw_settransformer_scheduler_probe | settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 0.6127 | 0.6082 | 15 |
| 17 | quickdraw_settransformer_lowshot_stable | 100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 0.6115 | 0.6084 | 19 |
| 18 | quickdraw_settransformer_stability | settransformer-sab/probes/lr5e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c500s | 0.6071 | 0.6047 | 15 |
| 19 | quickdraw_settransformer_stability | settransformer-sab/final/lr5e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 0.6061 | 0.6059 | 21 |
| 20 | quickdraw_settransformer_stability | settransformer-sab/probes/lr3e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c500s | 0.5891 | 0.5854 | 15 |

## Full Results by Experiment Group

### root quickdraw runs

- Files: **2**
- Best run: `deepset_single_quickdraw_e50` (deepset, test acc 0.7795, macro F1 0.7790)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| settransformer_isab_single_quickdraw | settransformer-isab | 10c920s | 9195 | 0.4402 | 0.4681 | 0.3876 | 0 |  | mps |  | xy=9; pool=sum | `settransformer_isab_single_quickdraw/summary.json` |
| deepset_single_quickdraw_e50 | deepset | 100c10000s | 1000000 | 0.7821 | 0.7795 | 0.7790 | 48 | 1667 | cuda | 1569 | bs=512; xy=9; pool=sum; padded=true | `deepset_single_quickdraw_e50/summary.json` |

### quickdraw 100c100s

- Files: **2**
- Best run: `deepset_single_quickdraw_e80` (deepset, test acc 0.4093, macro F1 0.3970)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| deepset_single_quickdraw_e50 | deepset | 100c100s | 10000 | 0.3940 | 0.3847 | 0.3779 | 37 | 18 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum; padded=true | `quickdraw_100c100s/deepset_single_quickdraw_e50/summary.json` |
| deepset_single_quickdraw_e80 | deepset | 100c100s | 10000 | 0.4060 | 0.4093 | 0.3970 | 28 | 67 | cuda | 1569 | limit=100; bs=64; xy=9; pool=sum; padded=true | `quickdraw_100c100s/deepset_single_quickdraw_e80/summary.json` |

### quickdraw deepset lowshot

- Files: **5**
- Best run: `100c1000s/deepset_single_quickdraw_e80` (deepset, test acc 0.6887, macro F1 0.6875)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/deepset_single_quickdraw_e80 | deepset | 100c50s | 5000 | 0.3280 | 0.3093 | 0.2774 | 21 | 15 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; padded=true | `quickdraw_deepset_lowshot/100c50s/deepset_single_quickdraw_e80/summary.json` |
| 100c100s/deepset_single_quickdraw_e80 | deepset | 100c100s | 10000 | 0.4027 | 0.4073 | 0.4000 | 76 | 29 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; padded=true | `quickdraw_deepset_lowshot/100c100s/deepset_single_quickdraw_e80/summary.json` |
| 100c200s/deepset_single_quickdraw_e80 | deepset | 100c200s | 20000 | 0.5403 | 0.5020 | 0.4981 | 26 | 66 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; padded=true | `quickdraw_deepset_lowshot/100c200s/deepset_single_quickdraw_e80/summary.json` |
| 100c500s/deepset_single_quickdraw_e80 | deepset | 100c500s | 50000 | 0.6231 | 0.6223 | 0.6209 | 22 | 148 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; padded=true | `quickdraw_deepset_lowshot/100c500s/deepset_single_quickdraw_e80/summary.json` |
| 100c1000s/deepset_single_quickdraw_e80 | deepset | 100c1000s | 100000 | 0.6825 | 0.6887 | 0.6875 | 46 | 235 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; padded=true | `quickdraw_deepset_lowshot/100c1000s/deepset_single_quickdraw_e80/summary.json` |

### quickdraw isab lowshot

- Files: **3**
- Best run: `100c200s/settransformer_isab_single_quickdraw_e80` (settransformer-isab, test acc 0.4073, macro F1 0.4092)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c50s | 5000 | 0.1440 | 0.1293 | 0.1251 | 75 | 98 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=mean; padded=true | `quickdraw_isab_lowshot/100c50s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c100s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c100s | 10000 | 0.0100 | 0.0073 | 0.0001 | 0 | 191 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=mean; padded=true | `quickdraw_isab_lowshot/100c100s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c200s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c200s | 20000 | 0.4237 | 0.4073 | 0.4092 | 57 | 489 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=mean; padded=true | `quickdraw_isab_lowshot/100c200s/settransformer_isab_single_quickdraw_e80/summary.json` |

### quickdraw sab lowshot

- Files: **3**
- Best run: `100c200s/settransformer_sab_single_quickdraw_e80` (settransformer-sab, test acc 0.4750, macro F1 0.4716)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c50s | 5000 | 0.2347 | 0.2467 | 0.2356 | 47 | 71 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=mean; padded=true | `quickdraw_sab_lowshot/100c50s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c100s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c100s | 10000 | 0.3527 | 0.3707 | 0.3655 | 62 | 241 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=mean; padded=true | `quickdraw_sab_lowshot/100c100s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c200s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c200s | 20000 | 0.5013 | 0.4750 | 0.4716 | 14 | 558 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=mean; padded=true | `quickdraw_sab_lowshot/100c200s/settransformer_sab_single_quickdraw_e80/summary.json` |

### quickdraw pointnet lowshot

- Files: **10**
- Best run: `100c1000s/pointnet_single_quickdraw_e80` (pointnet, test acc 0.7175, macro F1 0.7186)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/pointnet_single_quickdraw_e80 | pointnet | 100c50s | 5000 | 0.1933 | 0.1947 | 0.1789 | 66 | 158 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c50s/pointnet_single_quickdraw_e80/summary.json` |
| 100c50s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c50s | 5000 | 0.1293 | 0.1200 | 0.1114 | 25 | 98 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c50s/pointnet2_single_quickdraw_e80/summary.json` |
| 100c100s/pointnet_single_quickdraw_e80 | pointnet | 100c100s | 10000 | 0.3373 | 0.3640 | 0.3582 | 69 | 297 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c100s/pointnet_single_quickdraw_e80/summary.json` |
| 100c100s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c100s | 10000 | 0.3327 | 0.3387 | 0.3293 | 34 | 193 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c100s/pointnet2_single_quickdraw_e80/summary.json` |
| 100c200s/pointnet_single_quickdraw_e80 | pointnet | 100c200s | 20000 | 0.5783 | 0.5437 | 0.5419 | 78 | 630 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c200s/pointnet_single_quickdraw_e80/summary.json` |
| 100c200s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c200s | 20000 | 0.5547 | 0.5217 | 0.5201 | 73 | 383 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c200s/pointnet2_single_quickdraw_e80/summary.json` |
| 100c500s/pointnet_single_quickdraw_e80 | pointnet | 100c500s | 50000 | 0.6635 | 0.6773 | 0.6770 | 63 | 1544 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c500s/pointnet_single_quickdraw_e80/summary.json` |
| 100c500s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c500s | 50000 | 0.6571 | 0.6588 | 0.6562 | 78 | 947 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c500s/pointnet2_single_quickdraw_e80/summary.json` |
| 100c1000s/pointnet_single_quickdraw_e80 | pointnet | 100c1000s | 100000 | 0.7137 | 0.7175 | 0.7186 | 68 | 8683 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c1000s/pointnet_single_quickdraw_e80/summary.json` |
| 100c1000s/pointnet2_single_quickdraw_e80 | pointnet2 | 100c1000s | 100000 | 0.7055 | 0.7073 | 0.7066 | 59 | 1498 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_pointnet_lowshot/100c1000s/pointnet2_single_quickdraw_e80/summary.json` |

### quickdraw settransformer stability

- Files: **13**
- Best run: `settransformer-sab/final/lr5e-4_warm5_clip1/100c1000s/settransformer_sab_single_quickdraw_e80` (settransformer-sab, test acc 0.6841, macro F1 0.6816)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| settransformer-sab/final/lr5e-4_warm5_clip1/100c50s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c50s | 5000 | 0.1413 | 0.1240 | 0.0947 | 12 | 84 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/final/lr5e-4_warm5_clip1/100c50s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-sab/final/lr5e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c100s | 10000 | 0.2320 | 0.2360 | 0.2210 | 12 | 153 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/final/lr5e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-sab/probes/lr1e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c100s | 10000 | 0.1380 | 0.1467 | 0.1094 | 15 | 31 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=1.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr1e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/probes/lr3e-4_warm10_clip1/100c100s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c100s | 10000 | 0.2213 | 0.2260 | 0.2029 | 15 | 31 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=3.0e-04; warmup=10; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr3e-4_warm10_clip1/100c100s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/probes/lr3e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c100s | 10000 | 0.2273 | 0.2567 | 0.2377 | 14 | 31 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=3.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr3e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/probes/lr5e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c100s | 10000 | 0.2287 | 0.2487 | 0.2233 | 10 | 31 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr5e-4_warm5_clip1/100c100s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/final/lr5e-4_warm5_clip1/100c200s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c200s | 20000 | 0.4160 | 0.3977 | 0.3920 | 13 | 342 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/final/lr5e-4_warm5_clip1/100c200s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-sab/final/lr5e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 50000 | 0.6083 | 0.6061 | 0.6059 | 21 | 838 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/final/lr5e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-sab/probes/lr1e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c500s | 50000 | 0.4808 | 0.4825 | 0.4716 | 15 | 168 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=1.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr1e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/probes/lr3e-4_warm10_clip1/100c500s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c500s | 50000 | 0.5735 | 0.5800 | 0.5734 | 15 | 168 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=3.0e-04; warmup=10; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr3e-4_warm10_clip1/100c500s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/probes/lr3e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c500s | 50000 | 0.5760 | 0.5891 | 0.5854 | 15 | 169 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=3.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr3e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/probes/lr5e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16 | settransformer-sab | 100c500s | 50000 | 0.6009 | 0.6071 | 0.6047 | 15 | 169 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/probes/lr5e-4_warm5_clip1/100c500s/settransformer_sab_single_quickdraw_e16/summary.json` |
| settransformer-sab/final/lr5e-4_warm5_clip1/100c1000s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c1000s | 100000 | 0.6781 | 0.6841 | 0.6816 | 20 | 2376 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=constant; warmup=5; clip=1; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_stability/settransformer-sab/final/lr5e-4_warm5_clip1/100c1000s/settransformer_sab_single_quickdraw_e80/summary.json` |

### quickdraw settransformer multiseed

- Files: **9**
- Best run: `settransformer-sab/old_default/100c200s/seed1569/settransformer_sab_single_quickdraw_e40` (settransformer-sab, test acc 0.4080, macro F1 0.4059)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| settransformer-sab/old_default/100c100s/seed1569/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c100s | 10000 | 0.0087 | 0.0067 | 0.0001 | 0 | 79 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c100s/seed1569/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c100s/seed2024/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c100s | 10000 | 0.0240 | 0.0173 | 0.0009 | 3 | 79 | cuda | 2024 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c100s/seed2024/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c100s/seed3407/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c100s | 10000 | 0.0093 | 0.0053 | 0.0001 | 5 | 76 | cuda | 3407 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c100s/seed3407/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c100s/seed42/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c100s | 10000 | 0.0080 | 0.0093 | 0.0002 | 0 | 87 | cuda | 42 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c100s/seed42/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c100s/seed777/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c100s | 10000 | 0.2320 | 0.2200 | 0.2179 | 34 | 92 | cuda | 777 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c100s/seed777/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c200s/seed1569/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c200s | 20000 | 0.4197 | 0.4080 | 0.4059 | 27 | 177 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c200s/seed1569/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c200s/seed2024/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c200s | 20000 | 0.0087 | 0.0110 | 0.0002 | 0 | 157 | cuda | 2024 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c200s/seed2024/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c200s/seed3407/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c200s | 20000 | 0.0080 | 0.0107 | 0.0002 | 0 | 168 | cuda | 3407 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c200s/seed3407/settransformer_sab_single_quickdraw_e40/summary.json` |
| settransformer-sab/old_default/100c200s/seed42/settransformer_sab_single_quickdraw_e40 | settransformer-sab | 100c200s | 20000 | 0.0100 | 0.0063 | 0.0001 | 0 | 168 | cuda | 42 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `quickdraw_settransformer_multiseed/settransformer-sab/old_default/100c200s/seed42/settransformer_sab_single_quickdraw_e40/summary.json` |

### quickdraw settransformer quality probe

- Files: **12**
- Best run: `settransformer-sab/mean_pool_ls010/100c200s/settransformer_sab_single_quickdraw_e30` (settransformer-sab, test acc 0.4637, macro F1 0.4598)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| settransformer-sab/mean_pool_ls010/100c100s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c100s | 10000 | 0.3100 | 0.3153 | 0.3035 | 22 | 50 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=mean; ls=0.1; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/mean_pool_ls010/100c100s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/pma_ls015_dropout010/100c100s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c100s | 10000 | 0.2807 | 0.3020 | 0.2870 | 22 | 65 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.15; drop=0.1; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/pma_ls015_dropout010/100c100s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/baseline_cosine/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4080 | 0.3817 | 0.3743 | 17 | 137 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/baseline_cosine/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/label_smooth_005/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4257 | 0.4190 | 0.4141 | 14 | 140 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.05; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/label_smooth_005/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/label_smooth_010/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4507 | 0.4430 | 0.4319 | 13 | 135 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.1; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/label_smooth_010/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/label_smooth_015/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4653 | 0.4480 | 0.4386 | 13 | 133 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.15; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/label_smooth_015/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/mean_pool_ls010/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4883 | 0.4637 | 0.4598 | 26 | 114 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=mean; ls=0.1; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/mean_pool_ls010/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/mean_pool_sanity/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4513 | 0.4330 | 0.4315 | 19 | 112 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/mean_pool_sanity/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/pma_ls010_dropout010/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4660 | 0.4400 | 0.4340 | 20 | 145 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.1; drop=0.1; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/pma_ls010_dropout010/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/pma_ls015_dropout010/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.4723 | 0.4627 | 0.4569 | 25 | 147 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.15; drop=0.1; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/pma_ls015_dropout010/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/small64_1enc_2h/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.3650 | 0.3530 | 0.3398 | 28 | 74 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/small64_1enc_2h/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |
| settransformer-sab/small64_1enc_2h_ls005/100c200s/settransformer_sab_single_quickdraw_e30 | settransformer-sab | 100c200s | 20000 | 0.3770 | 0.3703 | 0.3544 | 27 | 78 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0.05; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_quality_probe/settransformer-sab/small64_1enc_2h_ls005/100c200s/settransformer_sab_single_quickdraw_e30/summary.json` |

### quickdraw settransformer scheduler probe

- Files: **6**
- Best run: `settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c500s/settransformer_sab_single_quickdraw_e80` (settransformer-sab, test acc 0.6127, macro F1 0.6082)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| settransformer-isab/lr5e-4_warm5_clip1_cosine_min1e-5/100c100s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c100s | 10000 | 0.3287 | 0.3113 | 0.2910 | 23 | 146 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_scheduler_probe/settransformer-isab/lr5e-4_warm5_clip1_cosine_min1e-5/100c100s/settransformer_isab_single_quickdraw_e80/summary.json` |
| settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c100s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c100s | 10000 | 0.2293 | 0.2327 | 0.2212 | 17 | 153 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_scheduler_probe/settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c100s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-isab/lr5e-4_warm5_clip1_cosine_min1e-5/100c200s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c200s | 20000 | 0.4807 | 0.4377 | 0.4346 | 36 | 308 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_scheduler_probe/settransformer-isab/lr5e-4_warm5_clip1_cosine_min1e-5/100c200s/settransformer_isab_single_quickdraw_e80/summary.json` |
| settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c200s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c200s | 20000 | 0.4297 | 0.4183 | 0.4110 | 13 | 341 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_scheduler_probe/settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c200s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-sab/lr5e-4_warm5_clip1_cosine_wd1e-5/100c200s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c200s | 20000 | 0.4140 | 0.3970 | 0.3884 | 13 | 341 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=1.0e-05; padded=true | `quickdraw_settransformer_scheduler_probe/settransformer-sab/lr5e-4_warm5_clip1_cosine_wd1e-5/100c200s/settransformer_sab_single_quickdraw_e80/summary.json` |
| settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 50000 | 0.6087 | 0.6127 | 0.6082 | 15 | 837 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_scheduler_probe/settransformer-sab/lr5e-4_warm5_clip1_cosine_min1e-5/100c500s/settransformer_sab_single_quickdraw_e80/summary.json` |

### quickdraw settransformer lowshot stable

- Files: **10**
- Best run: `100c1000s/settransformer_isab_single_quickdraw_e80` (settransformer-isab, test acc 0.7058, macro F1 0.7040)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c50s | 5000 | 0.1987 | 0.2000 | 0.1639 | 21 | 77 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c50s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c50s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c50s | 5000 | 0.1520 | 0.0960 | 0.0865 | 20 | 84 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c50s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c100s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c100s | 10000 | 0.3280 | 0.3220 | 0.3126 | 29 | 150 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c100s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c100s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c100s | 10000 | 0.2373 | 0.2513 | 0.2379 | 14 | 153 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c100s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c200s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c200s | 20000 | 0.4830 | 0.4517 | 0.4523 | 28 | 315 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c200s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c200s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c200s | 20000 | 0.4150 | 0.3953 | 0.3882 | 17 | 341 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c200s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c500s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c500s | 50000 | 0.6332 | 0.6357 | 0.6343 | 29 | 764 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c500s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 50000 | 0.6067 | 0.6115 | 0.6084 | 19 | 831 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c500s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c1000s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c1000s | 100000 | 0.7003 | 0.7058 | 0.7040 | 24 | 1667 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c1000s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c1000s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c1000s | 100000 | 0.6823 | 0.6867 | 0.6859 | 26 | 2881 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable/100c1000s/settransformer_sab_single_quickdraw_e80/summary.json` |

### quickdraw settransformer lowshot stable meanpool

- Files: **10**
- Best run: `100c1000s/settransformer_isab_single_quickdraw_e80` (settransformer-isab, test acc 0.6990, macro F1 0.6973)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c50s | 5000 | 0.2533 | 0.2467 | 0.2225 | 27 | 64 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c50s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c50s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c50s | 5000 | 0.1907 | 0.1520 | 0.1191 | 13 | 70 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c50s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c100s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c100s | 10000 | 0.3480 | 0.3527 | 0.3437 | 38 | 115 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c100s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c100s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c100s | 10000 | 0.2800 | 0.2940 | 0.2843 | 21 | 127 | cuda | 1569 | limit=100; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c100s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c200s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c200s | 20000 | 0.5173 | 0.4740 | 0.4747 | 22 | 247 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c200s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c200s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c200s | 20000 | 0.4473 | 0.4337 | 0.4264 | 17 | 285 | cuda | 1569 | limit=200; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c200s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c500s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c500s | 50000 | 0.6336 | 0.6325 | 0.6307 | 30 | 603 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c500s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c500s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c500s | 50000 | 0.6187 | 0.6191 | 0.6164 | 23 | 693 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c500s/settransformer_sab_single_quickdraw_e80/summary.json` |
| 100c1000s/settransformer_isab_single_quickdraw_e80 | settransformer-isab | 100c1000s | 100000 | 0.6965 | 0.6990 | 0.6973 | 27 | 1320 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c1000s/settransformer_isab_single_quickdraw_e80/summary.json` |
| 100c1000s/settransformer_sab_single_quickdraw_e80 | settransformer-sab | 100c1000s | 100000 | 0.6888 | 0.6962 | 0.6950 | 25 | 12384 | cuda | 1569 | limit=1000; bs=256; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `quickdraw_settransformer_lowshot_stable_meanpool/100c1000s/settransformer_sab_single_quickdraw_e80/summary.json` |

### quickdraw stability probe

- Files: **1**
- Best run: `sab_pma_lr3e4_warm5_clip1_100c500s/settransformer_sab_single_quickdraw_e12` (settransformer-sab, test acc 0.5403, macro F1 0.5358)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| sab_pma_lr3e4_warm5_clip1_100c500s/settransformer_sab_single_quickdraw_e12 | settransformer-sab | 100c500s | 50000 | 0.5509 | 0.5403 | 0.5358 | 11 | 127 | cuda | 1569 | limit=500; bs=128; xy=9; pool=sum; set_pool=pma; lr=3.0e-04; warmup=5; clip=1; wd=0; padded=true | `quickdraw_stability_probe/sab_pma_lr3e4_warm5_clip1_100c500s/settransformer_sab_single_quickdraw_e12/summary.json` |

### smoke quickdraw sweep

- Files: **3**
- Best run: `deepset_single_quickdraw_e1` (deepset, test acc 0.0067, macro F1 0.0003)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| deepset_single_quickdraw_e1 | deepset | 100c10s | 1000 | 0.0133 | 0.0067 | 0.0003 | 0 | 0 | cuda | 1569 | limit=10; bs=32; xy=9; pool=sum | `smoke_quickdraw_sweep/deepset_single_quickdraw_e1/summary.json` |
| settransformer_isab_single_quickdraw_e1 | settransformer-isab | 100c10s | 1000 | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | cuda | 1569 | limit=10; bs=32; xy=9; pool=sum | `smoke_quickdraw_sweep/settransformer_isab_single_quickdraw_e1/summary.json` |
| settransformer_sab_single_quickdraw_e1 | settransformer-sab | 100c10s | 1000 | 0.0133 | 0.0067 | 0.0002 | 0 | 1 | cuda | 1569 | limit=10; bs=32; xy=9; pool=sum | `smoke_quickdraw_sweep/settransformer_sab_single_quickdraw_e1/summary.json` |

### smoke quickdraw sweep b512

- Files: **3**
- Best run: `deepset_single_quickdraw_e1` (deepset, test acc 0.0100, macro F1 0.0023)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| deepset_single_quickdraw_e1 | deepset | 100c100s | 10000 | 0.0113 | 0.0100 | 0.0023 | 0 | 1 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum | `smoke_quickdraw_sweep_b512/deepset_single_quickdraw_e1/summary.json` |
| settransformer_isab_single_quickdraw_e1 | settransformer-isab | 100c100s | 10000 | 0.0100 | 0.0093 | 0.0007 | 0 | 4 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum | `smoke_quickdraw_sweep_b512/settransformer_isab_single_quickdraw_e1/summary.json` |
| settransformer_sab_single_quickdraw_e1 | settransformer-sab | 100c100s | 10000 | 0.0073 | 0.0087 | 0.0002 | 0 | 8 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum | `smoke_quickdraw_sweep_b512/settransformer_sab_single_quickdraw_e1/summary.json` |

### smoke quickdraw sweep b1024

- Files: **2**
- Best run: `settransformer_sab_single_quickdraw_e1` (settransformer-sab, test acc 0.0113, macro F1 0.0002)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| settransformer_isab_single_quickdraw_e1 | settransformer-isab | 100c100s | 10000 | 0.0093 | 0.0067 | 0.0005 | 0 | 16 | cuda | 1569 | limit=100; bs=1024; xy=9; pool=sum | `smoke_quickdraw_sweep_b1024/settransformer_isab_single_quickdraw_e1/summary.json` |
| settransformer_sab_single_quickdraw_e1 | settransformer-sab | 100c100s | 10000 | 0.0080 | 0.0113 | 0.0002 | 0 | 37 | cuda | 1569 | limit=100; bs=1024; xy=9; pool=sum | `smoke_quickdraw_sweep_b1024/settransformer_sab_single_quickdraw_e1/summary.json` |

### smoke quickdraw sweep padded b512

- Files: **3**
- Best run: `deepset_single_quickdraw_e1` (deepset, test acc 0.0100, macro F1 0.0023)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| deepset_single_quickdraw_e1 | deepset | 100c100s | 10000 | 0.0113 | 0.0100 | 0.0023 | 0 | 1 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum; padded=true | `smoke_quickdraw_sweep_padded_b512/deepset_single_quickdraw_e1/summary.json` |
| settransformer_isab_single_quickdraw_e1 | settransformer-isab | 100c100s | 10000 | 0.0113 | 0.0087 | 0.0007 | 0 | 4 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum; padded=true | `smoke_quickdraw_sweep_padded_b512/settransformer_isab_single_quickdraw_e1/summary.json` |
| settransformer_sab_single_quickdraw_e1 | settransformer-sab | 100c100s | 10000 | 0.0073 | 0.0087 | 0.0002 | 0 | 8 | cuda | 1569 | limit=100; bs=512; xy=9; pool=sum; padded=true | `smoke_quickdraw_sweep_padded_b512/settransformer_sab_single_quickdraw_e1/summary.json` |

### smoke quickdraw pointnet lowshot

- Files: **2**
- Best run: `100c50s/pointnet2_single_quickdraw_e1` (pointnet2, test acc 0.0173, macro F1 0.0023)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/pointnet_single_quickdraw_e1 | pointnet | 100c50s | 5000 | 0.0240 | 0.0133 | 0.0016 | 0 | 2 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `smoke_quickdraw_pointnet_lowshot/100c50s/pointnet_single_quickdraw_e1/summary.json` |
| 100c50s/pointnet2_single_quickdraw_e1 | pointnet2 | 100c50s | 5000 | 0.0227 | 0.0173 | 0.0023 | 0 | 1 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=0.001; sched=constant; warmup=0; clip=0; min_lr=0; wd=0; padded=true | `smoke_quickdraw_pointnet_lowshot/100c50s/pointnet2_single_quickdraw_e1/summary.json` |

### smoke quickdraw settransformer lowshot stable

- Files: **2**
- Best run: `100c50s/settransformer_sab_single_quickdraw_e1` (settransformer-sab, test acc 0.0093, macro F1 0.0002)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/settransformer_isab_single_quickdraw_e1 | settransformer-isab | 100c50s | 5000 | 0.0107 | 0.0067 | 0.0001 | 0 | 1 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `smoke_quickdraw_settransformer_lowshot_stable/100c50s/settransformer_isab_single_quickdraw_e1/summary.json` |
| 100c50s/settransformer_sab_single_quickdraw_e1 | settransformer-sab | 100c50s | 5000 | 0.0040 | 0.0093 | 0.0002 | 0 | 1 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=pma; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `smoke_quickdraw_settransformer_lowshot_stable/100c50s/settransformer_sab_single_quickdraw_e1/summary.json` |

### smoke quickdraw settransformer lowshot stable meanpool

- Files: **1**
- Best run: `100c50s/settransformer_sab_single_quickdraw_e1` (settransformer-sab, test acc 0.0093, macro F1 0.0002)

| Variant | Model | Subset | Samples | Val Acc | Test Acc | Macro F1 | Best Ep | Elapsed (s) | Device | Seed | Config | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 100c50s/settransformer_sab_single_quickdraw_e1 | settransformer-sab | 100c50s | 5000 | 0.0040 | 0.0093 | 0.0002 | 0 | 1 | cuda | 1569 | limit=50; bs=128; xy=9; pool=sum; set_pool=mean; ls=0; drop=0; lr=5.0e-04; sched=cosine; warmup=5; clip=1; min_lr=1.0e-05; wd=0; padded=true | `smoke_quickdraw_settransformer_lowshot_stable_meanpool/100c50s/settransformer_sab_single_quickdraw_e1/summary.json` |

