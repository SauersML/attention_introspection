| order | boosted | ×5 | ×10 | mass on word / label at ×10 (ratio to no boost) |
|---|---|---|---|---|
| inorder | word | +147% [+123, +175] | +580% [+492, +681] | 7.46× / 0.96× |
| inorder | label | +61% [+55, +68] | +144% [+130, +160] | 0.89× / 6.64× |
| inorder | both | +217% [+187, +252] | +674% [+581, +779] | 6.79× / 6.46× |
| shuffled | word | +12% [+5, +19] | +25% [+15, +35] | 7.56× / 0.96× |
| shuffled | label | +21% [+13, +30] | +36% [+22, +51] | 0.87× / 6.32× |
| shuffled | both | +22% [+11, +34] | +32% [+16, +51] | 6.65× / 6.14× |

All doses (specificity %):

| order | boosted | ×0.25 | ×0.5 | ×2 | ×5 | ×10 |
|---|---|---|---|---|---|---|
| inorder | word | -9% | -7% | +24% | +147% | +580% |
| inorder | label | -5% | -5% | +13% | +61% | +144% |
| inorder | both | -13% | -12% | +39% | +217% | +674% |
| shuffled | word | -1% | -1% | +5% | +12% | +25% |
| shuffled | label | -2% | -3% | +4% | +21% | +36% |
| shuffled | both | -3% | -4% | +7% | +22% | +32% |

Check, word target vs committed runs (specificity, log units):

- inorder: ×0.25: -0.092 (committed -0.092), ×0.5: -0.075 (committed -0.075), ×2.0: +0.216 (committed +0.216), ×5.0: +0.905 (committed +0.905), ×10.0: +1.917 (committed +1.917)
- shuffled: ×0.25: -0.008 (committed -0.008), ×0.5: -0.012 (committed -0.012), ×2.0: +0.051 (committed +0.051), ×5.0: +0.112 (committed +0.112), ×10.0: +0.220 (committed +0.220)

Paired differences (log units, 95% CI):

- inorder|label_minus_word|5.0: -0.429 [-0.543, -0.316], p 0
- shuffled|label_minus_word|5.0: +0.082 [-0.011, +0.174], p 0.083
- inorder_minus_shuffled|word|5.0: +0.794 [+0.703, +0.890], p 0
- inorder_minus_shuffled|label|5.0: +0.283 [+0.205, +0.362], p 0
- inorder_minus_shuffled|both|5.0: +0.955 [+0.859, +1.055], p 0
- inorder|label_minus_word|10.0: -1.024 [-1.174, -0.872], p 0
- shuffled|label_minus_word|10.0: +0.087 [-0.041, +0.212], p 0.17
- inorder_minus_shuffled|word|10.0: +1.697 [+1.557, +1.832], p 0
- inorder_minus_shuffled|label|10.0: +0.586 [+0.466, +0.707], p 0
- inorder_minus_shuffled|both|10.0: +1.765 [+1.610, +1.915], p 0

No-boost attention mass at the report-question queries (band, head and query mean): inorder: words 0.0082, labels 0.0243; shuffled: words 0.0079, labels 0.0305
