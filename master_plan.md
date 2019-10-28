# Master Plan

- [x] Prepare [re-map](https://github.com/aleksas/t2t-remap) tool for walking over corpora with sliding window after regex modifications.
- [x] Prepare pipeline for training and evaluating stressed LT texts
- [ ] Prepare pipeline for evaluating how many words were stressed correctly/incorrectly on a human stressed text.
- [ ] Reduce trained model instalbility
  - [ ] Train on other languages/content to produce output identical to input
  - [ ] Generate more stressed corpora using various combinations of automated stressing tools
    - [ ] Just t2t trained on semi-automatically stressed text ([1](https://github.com/aleksas/liepa_dataset/blob/master/other/stressed/chrestomatija.txt), [2](https://github.com/aleksas/liepa_dataset/blob/master/other/stressed/marti.txt), [3](https://github.com/aleksas/liepa_dataset/blob/master/other/stressed/__final_1.txt)).
    - [ ] Just [VDU SOAP Stressor](https://github.com/aleksas/denormalizer/blob/master/utils/vdu_soap_stressor.py)
    - [ ] Combination of [VDU SOAP Stressor](https://github.com/aleksas/denormalizer/blob/master/utils/vdu_soap_stressor.py) and [VDU annotator](https://github.com/aleksas/denormalizer/blob/master/utils/vdu_tagger.py)
