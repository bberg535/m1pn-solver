# Run times (tardis)

| task | device | nodes | moments | steps | wall s | solver s | estimate s |
|---|---|---:|---:|---:|---:|---:|---:|
| smooth_P3_p1_E8_spatial | CPU | 9 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p1_E16_spatial | CPU | 17 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p1_E32_spatial | CPU | 33 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p1_E64_spatial | CPU | 65 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p1_E128_spatial | CPU | 129 |  |  | 0.2 | 0.0 | 0 |
| smooth_P3_p1_E256_spatial | CPU | 257 |  |  | 0.3 | 0.0 | 0 |
| smoothdt2_P3_p1_E256_temporal_check | CPU | 257 |  |  | 0.4 | 0.0 | 0 |
| smooth_P3_p2_E8_spatial | CPU | 17 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p2_E16_spatial | CPU | 33 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p2_E32_spatial | CPU | 65 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p2_E64_spatial | CPU | 129 |  |  | 0.2 | 0.0 | 0 |
| smooth_P3_p2_E128_spatial | CPU | 257 |  |  | 0.3 | 0.0 | 0 |
| smooth_P3_p2_E256_spatial | CPU | 513 |  |  | 0.6 | 0.0 | 0 |
| smoothdt2_P3_p2_E256_temporal_check | CPU | 513 |  |  | 1.1 | 0.0 | 0 |
| smooth_P3_p3_E8_spatial | CPU | 25 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p3_E16_spatial | CPU | 49 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p3_E32_spatial | CPU | 97 |  |  | 0.2 | 0.0 | 0 |
| smooth_P3_p3_E64_spatial | CPU | 193 |  |  | 0.3 | 0.0 | 0 |
| smooth_P3_p3_E128_spatial | CPU | 385 |  |  | 0.6 | 0.0 | 0 |
| smooth_P3_p3_E256_spatial | CPU | 769 |  |  | 1.8 | 0.0 | 0 |
| smoothdt2_P3_p3_E256_temporal_check | CPU | 769 |  |  | 3.4 | 0.0 | 0 |
| smooth_P3_p4_E8_spatial | CPU | 33 |  |  | 0.1 | 0.0 | 0 |
| smooth_P3_p4_E16_spatial | CPU | 65 |  |  | 0.2 | 0.0 | 0 |
| smooth_P3_p4_E32_spatial | CPU | 129 |  |  | 0.2 | 0.0 | 0 |
| smooth_P3_p4_E64_spatial | CPU | 257 |  |  | 0.7 | 0.0 | 0 |
| smooth_P3_p4_E128_spatial | CPU | 513 |  |  | 2.7 | 0.0 | 0 |
| smooth_P3_p4_E256_spatial | CPU | 1025 |  |  | 12.6 | 0.0 | 0 |
| smoothdt2_P3_p4_E256_temporal_check | CPU | 1025 |  |  | 25.3 | 0.0 | 0 |
| boundary_P3_p2_E8_std | GPU | 289 | 11 | 39 | 2.2 | 0.0 | 0 |
| boundary_P3_p2_E16_std | GPU | 1089 | 11 | 78 | 2.3 | 0.1 | 1 |
| boundary_P3_p2_E32_std | GPU | 4225 | 11 | 155 | 2.5 | 0.2 | 1 |
| boundary_P3_p2_E64_std | GPU | 16641 | 11 | 310 | 3.1 | 0.6 | 2 |
| boundary_P3_p2_E128_std | GPU | 66049 | 11 | 619 | 5.2 | 2.1 | 5 |
| boundary_P3_p2_E256_std | GPU | 263169 | 11 | 1235 | 14.5 | 10.0 | 14 |
| boundary_P3_p2_E512_std | GPU | 1050625 | 11 | 2467 | 70.1 | 60.0 | 55 |
| boundary_P7_p2_E8_std | GPU | 289 | 37 | 40 | 2.2 | 0.0 | 0 |
| boundary_P7_p2_E16_std | GPU | 1089 | 37 | 78 | 2.3 | 0.1 | 1 |
| boundary_P7_p2_E32_std | GPU | 4225 | 37 | 156 | 2.6 | 0.3 | 1 |
| boundary_P7_p2_E64_std | GPU | 16641 | 37 | 310 | 3.6 | 1.0 | 3 |
| boundary_P7_p2_E128_std | GPU | 66049 | 37 | 619 | 7.2 | 4.2 | 7 |
| boundary_P7_p2_E256_std | GPU | 263169 | 37 | 1236 | 24.1 | 19.3 | 25 |
| boundary_P7_p2_E512_std | GPU | 1050625 | 37 | 2468 | 124.1 | 112.8 | 141 |
| boundary_P11_p2_E8_std | GPU | 289 | 79 | 40 | 2.3 | 0.1 | 0 |
| boundary_P11_p2_E16_std | GPU | 1089 | 79 | 79 | 2.5 | 0.2 | 1 |
| boundary_P11_p2_E32_std | GPU | 4225 | 79 | 156 | 3.1 | 0.7 | 1 |
| boundary_P11_p2_E64_std | GPU | 16641 | 79 | 310 | 5.0 | 2.4 | 3 |
| boundary_P11_p2_E128_std | GPU | 66049 | 79 | 619 | 13.5 | 10.2 | 9 |
| boundary_P11_p2_E256_std | GPU | 263169 | 79 | 1235 | 51.4 | 45.6 | 42 |
| boundary_P11_p2_E512_std | GPU | 1050625 | 79 | 2468 | 260.2 | 246.4 | 281 |
| boundary_P15_p2_E8_std | GPU | 289 | 137 | 39 | 2.5 | 0.1 | 0 |
| boundary_P15_p2_E16_std | GPU | 1089 | 137 | 78 | 2.9 | 0.4 | 1 |
| boundary_P15_p2_E32_std | GPU | 4225 | 137 | 156 | 4.1 | 1.6 | 1 |
| boundary_P15_p2_E64_std | GPU | 16641 | 137 | 310 | 8.6 | 5.8 | 3 |
| boundary_P15_p2_E128_std | GPU | 66049 | 137 | 619 | 27.8 | 24.0 | 12 |
| boundary_P15_p2_E256_std | GPU | 263169 | 137 | 1235 | 111.4 | 104.6 | 66 |
| boundary_P15_p2_E512_std | GPU | 1050625 | 137 | 2468 | 529.9 | 512.9 | 474 |
| crossing_P3_p1_E16_std | GPU | 289 | 11 | 102 | 2.3 | 0.1 | 1 |
| crossing_P3_p1_E32_std | GPU | 1089 | 11 | 203 | 2.7 | 0.3 | 1 |
| crossing_P3_p1_E64_std | GPU | 4225 | 11 | 406 | 3.3 | 0.8 | 2 |
| crossing_P3_p1_E128_std | GPU | 16641 | 11 | 812 | 4.9 | 2.0 | 5 |
| crossing_P3_p1_E256_std | GPU | 66049 | 11 | 1623 | 9.9 | 6.0 | 11 |
| crossing_P3_p1_E512_std | GPU | 263169 | 11 | 3246 | 33.4 | 26.9 | 29 |
| crossing_P3_p1_E1024_std | GPU | 1050625 | 11 | 6492 | 930.0 | 916.0 | 117 |
| crossing_P3_p1_E2048_std | GPU | 4198401 |  |  | 4213.9 | 0.0 | 702 |
| crossing_P3_p2_E8_std | GPU | 289 | 11 | 84 | 2.4 | 0.2 | 1 |
| crossing_P3_p2_E16_std | GPU | 1089 | 11 | 166 | 2.7 | 0.4 | 1 |
| crossing_P3_p2_E32_std | GPU | 4225 | 11 | 331 | 3.5 | 1.1 | 2 |
| crossing_P3_p2_E64_std | GPU | 16641 | 11 | 661 | 5.5 | 2.7 | 5 |
| crossing_P3_p2_E128_std | GPU | 66049 | 11 | 1320 | 11.3 | 7.7 | 11 |
| crossing_P3_p2_E256_std | GPU | 263169 | 11 | 2634 | 35.5 | 29.4 | 29 |
| crossing_P3_p2_E512_std | GPU | 1050625 | 11 | 5263 | 183.3 | 170.3 | 117 |
| crossing_P3_p3_E8_std | GPU | 625 | 11 | 157 | 3.0 | 0.7 | 1 |
| crossing_P3_p3_E16_std | GPU | 2401 | 11 | 314 | 4.0 | 1.6 | 2 |
| crossing_P3_p3_E32_std | GPU | 9409 | 11 | 627 | 6.7 | 4.1 | 4 |
| crossing_P3_p3_E64_std | GPU | 37249 | 11 | 1251 | 13.8 | 10.4 | 8 |
| crossing_P3_p3_E128_std | GPU | 148225 | 11 | 2496 | 37.8 | 32.7 | 19 |
| crossing_P3_p3_E256_std | GPU | 591361 | 11 | 4982 | 124.5 | 114.5 | 62 |
| crossing_P7_p1_E16_std | GPU | 289 | 37 | 102 | 2.4 | 0.1 | 1 |
| crossing_P7_p1_E32_std | GPU | 1089 | 37 | 203 | 2.6 | 0.3 | 1 |
| crossing_P7_p1_E64_std | GPU | 4225 | 37 | 406 | 3.1 | 0.6 | 3 |
| crossing_P7_p1_E128_std | GPU | 16641 | 37 | 812 | 4.7 | 1.8 | 5 |
| crossing_P7_p1_E256_std | GPU | 66049 | 37 | 1623 | 10.9 | 7.0 | 14 |
| crossing_P7_p1_E512_std | GPU | 263169 | 37 | 3246 | 43.9 | 36.6 | 53 |
| crossing_P7_p1_E1024_std | GPU | 1050625 | 37 | 6492 | 287.1 | 271.4 | 302 |
| crossing_P7_p1_E2048_std | GPU | 4198401 |  |  | 13056.1 | 0.0 | 2176 |
| crossing_P7_p2_E8_std | GPU | 289 | 37 | 84 | 2.4 | 0.2 | 1 |
| crossing_P7_p2_E16_std | GPU | 1089 | 37 | 166 | 2.7 | 0.4 | 1 |
| crossing_P7_p2_E32_std | GPU | 4225 | 37 | 332 | 3.2 | 0.7 | 3 |
| crossing_P7_p2_E64_std | GPU | 16641 | 37 | 661 | 4.9 | 2.1 | 5 |
| crossing_P7_p2_E128_std | GPU | 66049 | 37 | 1320 | 11.3 | 7.7 | 14 |
| crossing_P7_p2_E256_std | GPU | 263169 | 37 | 2635 | 44.3 | 37.6 | 53 |
| crossing_P7_p2_E512_std | GPU | 1050625 | 37 | 5264 | 274.3 | 258.7 | 302 |
| crossing_P7_p3_E8_std | GPU | 625 | 37 | 158 | 3.0 | 0.7 | 1 |
| crossing_P7_p3_E16_std | GPU | 2401 | 37 | 315 | 4.0 | 1.5 | 2 |
| crossing_P7_p3_E32_std | GPU | 9409 | 37 | 628 | 6.0 | 3.3 | 4 |
| crossing_P7_p3_E64_std | GPU | 37249 | 37 | 1252 | 11.5 | 8.0 | 9 |
| crossing_P7_p3_E128_std | GPU | 148225 | 37 | 2496 | 37.4 | 31.6 | 29 |
| crossing_P7_p3_E256_std | GPU | 591361 | 37 | 4982 | 172.8 | 161.9 | 140 |
| crossing_P11_p1_E16_std | GPU | 289 | 79 | 102 | 2.4 | 0.2 | 1 |
| crossing_P11_p1_E32_std | GPU | 1089 | 79 | 203 | 2.6 | 0.3 | 1 |
| crossing_P11_p1_E64_std | GPU | 4225 | 79 | 406 | 3.3 | 0.8 | 3 |
| crossing_P11_p1_E128_std | GPU | 16641 | 79 | 812 | 5.2 | 2.2 | 6 |
| crossing_P11_p1_E256_std | GPU | 66049 | 79 | 1623 | 14.3 | 10.2 | 19 |
| crossing_P11_p1_E512_std | GPU | 263169 | 79 | 3246 | 66.6 | 58.6 | 90 |
| crossing_P11_p1_E1024_std | GPU | 1050625 | 79 | 6492 | 467.9 | 448.9 | 600 |
| crossing_P11_p2_E8_std | GPU | 289 | 79 | 84 | 2.5 | 0.2 | 1 |
| crossing_P11_p2_E16_std | GPU | 1089 | 79 | 167 | 2.8 | 0.4 | 1 |
| crossing_P11_p2_E32_std | GPU | 4225 | 79 | 332 | 3.3 | 0.8 | 3 |
| crossing_P11_p2_E64_std | GPU | 16641 | 79 | 661 | 5.2 | 2.3 | 6 |
| crossing_P11_p2_E128_std | GPU | 66049 | 79 | 1320 | 14.3 | 10.4 | 19 |
| crossing_P11_p2_E256_std | GPU | 263169 | 79 | 2635 | 64.1 | 56.8 | 90 |
| crossing_P11_p2_E512_std | GPU | 1050625 | 79 | 5264 | 441.4 | 423.8 | 600 |
| crossing_P11_p3_E8_std | GPU | 625 | 79 | 159 | 3.1 | 0.7 | 1 |
| crossing_P11_p3_E16_std | GPU | 2401 | 79 | 315 | 4.0 | 1.5 | 2 |
| crossing_P11_p3_E32_std | GPU | 9409 | 79 | 628 | 5.9 | 3.2 | 4 |
| crossing_P11_p3_E64_std | GPU | 37249 | 79 | 1252 | 11.9 | 8.4 | 11 |
| crossing_P11_p3_E128_std | GPU | 148225 | 79 | 2497 | 49.7 | 43.6 | 44 |
| crossing_P11_p3_E256_std | GPU | 591361 | 79 | 4982 | 281.4 | 268.6 | 266 |
| crossing_P15_p1_E16_std | GPU | 289 | 137 | 102 | 2.7 | 0.2 | 1 |
| crossing_P15_p1_E32_std | GPU | 1089 | 137 | 203 | 3.0 | 0.5 | 1 |
| crossing_P15_p1_E64_std | GPU | 4225 | 137 | 406 | 4.4 | 1.7 | 3 |
| crossing_P15_p1_E128_std | GPU | 16641 | 137 | 812 | 7.2 | 3.9 | 7 |
| crossing_P15_p1_E256_std | GPU | 66049 | 137 | 1623 | 21.5 | 16.9 | 25 |
| crossing_P15_p1_E512_std | GPU | 263169 | 137 | 3246 | 102.2 | 93.0 | 141 |
| crossing_P15_p1_E1024_std | GPU | 1050625 | 137 | 6492 | 738.0 | 714.8 | 1011 |
| crossing_P15_p2_E8_std | GPU | 289 | 137 | 84 | 2.6 | 0.2 | 1 |
| crossing_P15_p2_E16_std | GPU | 1089 | 137 | 167 | 3.0 | 0.5 | 1 |
| crossing_P15_p2_E32_std | GPU | 4225 | 137 | 332 | 4.1 | 1.4 | 3 |
| crossing_P15_p2_E64_std | GPU | 16641 | 137 | 662 | 6.4 | 3.3 | 7 |
| crossing_P15_p2_E128_std | GPU | 66049 | 137 | 1320 | 20.7 | 16.3 | 25 |
| crossing_P15_p2_E256_std | GPU | 263169 | 137 | 2635 | 95.9 | 87.6 | 141 |
| crossing_P15_p2_E512_std | GPU | 1050625 | 137 | 5264 | 688.6 | 666.5 | 1011 |
| crossing_P15_p3_E8_std | GPU | 625 | 137 | 158 | 3.2 | 0.7 | 1 |
| crossing_P15_p3_E16_std | GPU | 2401 | 137 | 316 | 4.3 | 1.7 | 2 |
| crossing_P15_p3_E32_std | GPU | 9409 | 137 | 629 | 6.4 | 3.4 | 5 |
| crossing_P15_p3_E64_std | GPU | 37249 | 137 | 1252 | 15.3 | 11.4 | 14 |
| crossing_P15_p3_E128_std | GPU | 148225 | 137 | 2498 | 62.4 | 55.3 | 66 |
| crossing_P15_p3_E256_std | GPU | 591361 | 137 | 4983 | 421.9 | 406.1 | 440 |
| line_P3_p1_E32_std | GPU | 1089 | 11 | 212 | 2.6 | 0.3 | 1 |
| line_P3_p1_E64_std | GPU | 4225 | 11 | 423 | 3.1 | 0.7 | 3 |
| line_P3_p1_E128_std | GPU | 16641 | 11 | 846 | 4.6 | 1.7 | 5 |
| line_P3_p1_E256_std | GPU | 66049 | 11 | 1691 | 9.0 | 5.2 | 12 |
| line_P3_p1_E512_std | GPU | 263169 | 11 | 3382 | 29.8 | 23.4 | 31 |
| line_P3_p1_E1024_std | GPU | 1050625 | 11 | 6763 | 1189.0 | 1175.4 | 122 |
| line_P3_p2_E16_std | GPU | 1089 | 11 | 173 | 2.6 | 0.3 | 1 |
| line_P3_p2_E32_std | GPU | 4225 | 11 | 345 | 3.2 | 0.7 | 3 |
| line_P3_p2_E64_std | GPU | 16641 | 11 | 689 | 4.8 | 2.1 | 5 |
| line_P3_p2_E128_std | GPU | 66049 | 11 | 1375 | 10.1 | 6.5 | 12 |
| line_P3_p2_E256_std | GPU | 263169 | 11 | 2743 | 33.0 | 26.6 | 31 |
| line_P3_p2_E512_std | GPU | 1050625 | 11 | 5482 | 173.4 | 161.0 | 122 |
| line_P7_p1_E32_std | GPU | 1089 | 37 | 212 | 2.6 | 0.3 | 1 |
| line_P7_p1_E64_std | GPU | 4225 | 37 | 423 | 3.2 | 0.7 | 3 |
| line_P7_p1_E128_std | GPU | 16641 | 37 | 846 | 4.8 | 2.0 | 6 |
| line_P7_p1_E256_std | GPU | 66049 | 37 | 1691 | 11.2 | 7.3 | 15 |
| line_P7_p1_E512_std | GPU | 263169 | 37 | 3382 | 44.9 | 37.7 | 55 |
| line_P7_p1_E1024_std | GPU | 1050625 | 37 | 6763 | 1056.4 | 1041.4 | 314 |
| line_P7_p2_E16_std | GPU | 1089 | 37 | 173 | 2.6 | 0.3 | 1 |
| line_P7_p2_E32_std | GPU | 4225 | 37 | 346 | 3.2 | 0.7 | 3 |
| line_P7_p2_E64_std | GPU | 16641 | 37 | 689 | 5.0 | 2.2 | 6 |
| line_P7_p2_E128_std | GPU | 66049 | 37 | 1375 | 11.9 | 8.3 | 15 |
| line_P7_p2_E256_std | GPU | 263169 | 37 | 2745 | 45.6 | 38.8 | 55 |
| line_P7_p2_E512_std | GPU | 1050625 | 37 | 5483 | 287.8 | 273.6 | 314 |
| line_P11_p1_E32_std | GPU | 1089 | 79 | 212 | 2.7 | 0.3 | 1 |
| line_P11_p1_E64_std | GPU | 4225 | 79 | 423 | 3.3 | 0.8 | 3 |
| line_P11_p1_E128_std | GPU | 16641 | 79 | 846 | 5.5 | 2.5 | 6 |
| line_P11_p1_E256_std | GPU | 66049 | 79 | 1691 | 15.5 | 11.4 | 19 |
| line_P11_p1_E512_std | GPU | 263169 | 79 | 3382 | 74.0 | 66.4 | 94 |
| line_P11_p1_E1024_std | GPU | 1050625 | 79 | 6763 | 1357.8 | 1340.9 | 625 |
| line_P11_p2_E16_std | GPU | 1089 | 79 | 174 | 2.6 | 0.3 | 1 |
| line_P11_p2_E32_std | GPU | 4225 | 79 | 346 | 3.2 | 0.7 | 3 |
| line_P11_p2_E64_std | GPU | 16641 | 79 | 689 | 5.5 | 2.7 | 6 |
| line_P11_p2_E128_std | GPU | 66049 | 79 | 1375 | 15.6 | 11.8 | 19 |
| line_P11_p2_E256_std | GPU | 263169 | 79 | 2745 | 70.6 | 63.6 | 94 |
| line_P11_p2_E512_std | GPU | 1050625 | 79 | 5483 | 488.3 | 472.3 | 625 |
| line_P15_p1_E32_std | GPU | 1089 | 137 | 212 | 3.1 | 0.6 | 1 |
| line_P15_p1_E64_std | GPU | 4225 | 137 | 423 | 4.6 | 1.8 | 3 |
| line_P15_p1_E128_std | GPU | 16641 | 137 | 846 | 7.3 | 4.1 | 7 |
| line_P15_p1_E256_std | GPU | 66049 | 137 | 1691 | 26.3 | 21.9 | 26 |
| line_P15_p1_E512_std | GPU | 263169 | 137 | 3382 | 117.2 | 109.1 | 147 |
| line_P15_p1_E1024_std | GPU | 1050625 | 137 | 6763 | 1397.4 | 1377.3 | 1053 |
| line_P15_p2_E16_std | GPU | 1089 | 137 | 174 | 3.0 | 0.5 | 1 |
| line_P15_p2_E32_std | GPU | 4225 | 137 | 346 | 4.8 | 2.1 | 3 |
| line_P15_p2_E64_std | GPU | 16641 | 137 | 689 | 6.8 | 3.7 | 7 |
| line_P15_p2_E128_std | GPU | 66049 | 137 | 1375 | 23.4 | 19.2 | 26 |
| line_P15_p2_E256_std | GPU | 263169 | 137 | 2744 | 109.7 | 102.1 | 147 |
| line_P15_p2_E512_std | GPU | 1050625 | 137 | 5483 | 792.3 | 772.9 | 1053 |
| time_smooth_P3_p2_E96_dt0 | CPU | 193 | 8 | 125 | 0.3 | 0.1 | 0 |
| time_smooth_P3_p2_E96_dt1 | CPU | 193 | 8 | 248 | 0.5 | 0.2 | 0 |
| time_smooth_P3_p2_E96_dt2 | CPU | 193 | 8 | 496 | 0.9 | 0.4 | 0 |
| time_smooth_P3_p2_E96_dt3 | CPU | 193 | 8 | 992 | 1.5 | 0.7 | 0 |
| time_smooth_P3_p2_E96_dt4 | CPU | 193 | 8 | 1983 | 3.0 | 1.5 | 0 |
| time_crossing_P7_p2_E128_dt0 | GPU | 66049 | 37 | 221 | 3.8 | 1.0 | 2 |
| time_crossing_P7_p2_E128_dt1 | GPU | 66049 | 37 | 440 | 5.0 | 2.0 | 2 |
| time_crossing_P7_p2_E128_dt2 | GPU | 66049 | 37 | 881 | 7.3 | 4.0 | 2 |
| time_crossing_P7_p2_E128_dt3 | GPU | 66049 | 37 | 1760 | 12.0 | 8.0 | 2 |
| time_crossing_P7_p2_E128_dt4 | GPU | 66049 | 37 | 3519 | 21.3 | 15.9 | 2 |
| ablation_P7_p2_E256_full | GPU | 263169 | 37 | 2635 | 44.1 | 37.6 | 53 |
| ablation_P7_p2_E256_no_weno | GPU | 263169 | 37 | 2635 | 43.6 | 36.9 | 53 |
| ablation_P7_p2_E256_no_limit | GPU | 263169 |  |  | 14.9 | 0.0 | 53 |
| ablation_P7_p2_E256_ssprk3 | GPU | 263169 | 37 | 5400 | 66.7 | 57.8 | 95 |
| ablation_P7_p2_E256_weno_weight_02 | GPU | 263169 | 37 | 2635 | 44.2 | 37.6 | 53 |
