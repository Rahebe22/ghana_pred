# Cat's logging
04/28/22: Run in 'DL/models/gh_cg_tz/SimpleDFUNet_04282022'. Model:
- SimpleDFUNet, 100 epochs
- img_path_cols = ['dir_os'] #train in one season
- img_bands = 4 instead of 8
- running train, got (-215:Assertion failed_step >= minstep in function 'cv::Mat::Mat'
- fixed by brightness_shift_subsets = [4] instead of [4,4]
- train dataloader took 6815s, saved as train_data_os.pkl and validate_data_os.pkl for validate loader

05/03/22: Reran predictions for SimpleDFUNet and DFUNet without attention (DFUNet 2).
- Can't currently run predictions for DFUNet with attention (possibly bc of difference in ErrCorrBlock)
- Todo: add separate ErrCorrBlock code (without importing) for each of the DFUNet with and without attention 

05/04/22: Ran predictions for DFUNet with attention (DFUNet 1) by uncommenting out = self.se(out) in ErrCorrBlock
- Move ErrCorrBlock code to separate models

05/17-18/22: Ran predictions on colab
- For DFUNet_noattn, comment out self.se(out) in `forward` in the error correction block
- For SimpleDFUNet, which was trained on `os` off season only, also apply the same changes (4 bands instead of 8, etc like detailed in 04/28)

06/13/22: Refine SimpleDFUNet models
- Trained on off season only
- Catalog: `catalog_gh_cg_tz_0-3500_2-8_3-1141_4-108_aois.csv`
- Epoch: 25 if len(data) < 300, else 50
- `learning_rate_init` = 0.0003
- `train_batch` = 32 `validate_batch` = 15
- Filter by aoi, 1-16
- Use a subset as validation data, 30 or 10% of data rounded, whichever is larger
- Uploaded final params to `DL/models/gh_cg_tz/SimpleDFUNet_refine_06132022/`

06/16/22: New refined SimpleDFUNet:
- Everything same except Epoch: 30 if len(data) < 300, else 50
- Using only label group 3 and 4 (if aoi has it) for validation
- Except aoi 16, epoch 30, learning_rate_init = 0.001

06/25/2022: Refine tanzania on SimpleDFUNet
- learning_rate_init = 0.001, epoch = 100, 578 labels in total
- same catalog in 6/13
- Uploaded final params to `DL/models/gh_cg_tz/SimpleDFUNet_04282022/SimpleDFUNet_params.pth`

07/12/2022: Refine SimpleDFUnet, last layer frozen
