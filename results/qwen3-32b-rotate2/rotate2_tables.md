# Rotation with focus and completely-ignore (Qwen3-32B, 100 arithmetic contexts, layers 21–53, answer-token queries, teacher-forced on the no-instruction answer)

## focus: % change (rows named, columns measured), mean over contexts [95% bootstrap CI]

| named \ measured | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | **+69.2 [+65.4, +73.2]** | +10.8 [+9.6, +12.0] | +0.3 [−1.0, +1.8] | −5.7 [−6.6, −4.8] |
| P2 | +21.1 [+19.5, +22.7] | **+50.0 [+46.8, +53.3]** | +18.9 [+17.3, +20.9] | +2.2 [+1.2, +3.2] |
| P3 | +7.5 [+6.0, +8.9] | +23.7 [+21.6, +25.9] | **+46.9 [+44.3, +49.7]** | +17.4 [+16.0, +18.9] |
| P4 | −7.0 [−8.1, −6.0] | +1.4 [+0.5, +2.3] | +12.1 [+10.7, +13.8] | **+35.1 [+33.2, +36.9]** |

Share of (context, named paragraph) pairs where the named paragraph moved most in the instructed direction: 98.0% [96.5, 99.2] (chance 25%); by named P1..P4 100%, 97%, 97%, 98%; contexts with 0..4 of 4: [0, 0, 0, 8, 92]; diagonal in instructed direction: 100%, 100%, 100%, 100%

## ignore: % change (rows named, columns measured), mean over contexts [95% bootstrap CI]

| named \ measured | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | **−13.5 [−14.5, −12.4]** | −5.7 [−6.5, −4.9] | −4.2 [−4.9, −3.4] | −4.6 [−5.5, −3.8] |
| P2 | −8.5 [−9.5, −7.5] | **−15.3 [−16.4, −14.3]** | −8.5 [−9.3, −7.6] | −7.9 [−8.8, −7.0] |
| P3 | −4.1 [−5.1, −3.2] | −9.1 [−10.0, −8.2] | **−14.9 [−15.8, −14.0]** | −10.4 [−11.3, −9.5] |
| P4 | +0.5 [−0.5, +1.5] | −3.7 [−4.6, −2.9] | −6.0 [−6.8, −5.1] | **−15.8 [−16.6, −14.9]** |

Share of (context, named paragraph) pairs where the named paragraph moved most in the instructed direction: 81.8% [77.5, 85.8] (chance 25%); by named P1..P4 83%, 75%, 72%, 97%; contexts with 0..4 of 4: [0, 5, 11, 36, 48]; diagonal in instructed direction: 100%, 100%, 100%, 100%

## focus_vs_none: % change (rows named, columns measured), mean over contexts [95% bootstrap CI]

| named \ measured | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | **+84.9 [+79.0, +91.5]** | +17.4 [+15.9, +19.0] | −1.7 [−3.2, +0.0] | −21.7 [−22.9, −20.4] |
| P2 | +26.8 [+24.1, +29.7] | **+83.5 [+77.5, +89.8]** | +25.0 [+22.6, +27.7] | −13.0 [−14.5, −11.5] |
| P3 | +0.5 [−1.4, +2.6] | +38.7 [+35.4, +42.3] | **+71.8 [+67.5, +76.2]** | +7.2 [+5.1, +9.4] |
| P4 | −18.8 [−20.3, −17.4] | +1.1 [−0.2, +2.5] | +12.0 [+10.3, +13.9] | **+27.7 [+25.8, +29.7]** |

Share of (context, named paragraph) pairs where the named paragraph moved most in the instructed direction: 97.0% [95.2, 98.5] (chance 25%); by named P1..P4 100%, 100%, 92%, 96%; contexts with 0..4 of 4: [0, 0, 0, 12, 88]; diagonal in instructed direction: 100%, 100%, 100%, 100%

## ignore_vs_none: % change (rows named, columns measured), mean over contexts [95% bootstrap CI]

| named \ measured | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | **−6.1 [−7.1, −5.0]** | −0.1 [−0.9, +0.8] | −6.1 [−7.0, −5.3] | −20.9 [−21.8, −19.9] |
| P2 | −4.5 [−5.5, −3.4] | **+2.7 [+1.7, +3.6]** | −4.0 [−4.9, −3.1] | −21.8 [−22.6, −20.8] |
| P3 | −10.6 [−11.4, −9.8] | +1.5 [+0.7, +2.4] | **−1.0 [−1.9, −0.0]** | −18.4 [−19.3, −17.5] |
| P4 | −12.5 [−13.4, −11.5] | −4.1 [−4.9, −3.3] | −6.1 [−7.0, −5.3] | **−20.4 [−21.3, −19.5]** |

Share of (context, named paragraph) pairs where the named paragraph moved most in the instructed direction: 23.0% [21.5, 24.2] (chance 25%); by named P1..P4 3%, 0%, 1%, 88%; contexts with 0..4 of 4: [8, 92, 0, 0, 0]; diagonal in instructed direction: 86%, 34%, 57%, 99%

Free (greedy) answers identical to the no-instruction answer: mention 99.0%, focus 99.5%, ignore 99.2%; no-instruction answer correct 99%.