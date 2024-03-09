# Run Semantic Segmentation on Agriculture Mapping Data

Boka Luo

09/01/2020



This is a tutorial on running semantic segmentation with package `deeplearner`for the Agriculture Mapping research project. The package by far provides models of:

* [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
* [DeepLab v3](https://arxiv.org/pdf/1706.05587.pdf)
* [DeepLab v3+](https://arxiv.org/pdf/1802.02611.pdf)
* [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/pdf/1612.01105.pdf)
* [ExFuse](https://arxiv.org/pdf/1804.03821.pdf)
* [Global Convolutional Network (GCN)](https://arxiv.org/pdf/1703.02719.pdf)



## Prerequisite

### Python Packages

To start, please make sure these packages are installed: yaml, numpy, pandas, rasterio, skimage, sklearn, pytorch, tensorboardX.

### Data

Based on current protocol, the project uses two image composites of 2022 × 2022built separately from growing-season and off-season time series, which are in size of a tile ( 2000 × 2000) plus a buffer of 11 pixels on each side. However, the labels are in grid size of 200 x 200, and was sorted into 4 groups:

* label_group = 0 -- labels that are not reviewed

* label_group = 2 -- labels have both positive and negative categories, while the correctly classified positive category is between 65% and 80%

* label_group = 3 -- labels have both positive and negative categories, while the correctly classified positive category is over 80%

* label_group = 4 -- labels have only negative categories, but it's overall accuracy is 100%



The `deeplearner` package uses csv files to load data. Therefore, two catalogs are required besides the raw images and labels. One is for train and validation, while another one is for prediction:

* catalog for train and validation

  * It contains at least 4 groups of columns:

    * columns for image directories, could be either a relative path to a data folder, or a full path in aws s3, starting with `s3://`
    * a column for label directories, could be either a relative path to a data folder, or a full path in aws s3, starting with `s3://`
    * a column named `usage`, where the usage value is `train` or `validate`
    * a column named `label_group `

  * Here‘s an example of the table format, where `dir_gs` and `dir_os` are directories to images and `dir_label` is directories to labels

    | name      | usage    | dir_gs                                               | dir_os                                               | dir_label                                                    | label_group |
    | --------- | -------- | ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ | ----------- |
    | GH0242195 | train    | images/planet/nonfix/GS/tile539785_736815_736967.tif | images/planet/nonfix/OS/tile539785_737029_737118.tif | labels/semantic_segmentation/accurate/GH0242195_3241_5699.tif | 3           |
    | GH0288657 | validate | images/planet/nonfix/GS/tile539959_736815_736967.tif | images/planet/nonfix/OS/tile539959_737029_737118.tif | labels/semantic_segmentation/accurate/GH0288657_3385_5774.tif | 3           |

* catalog for prediction

  * It contains at least 3 groups of columns:

    * columns for image directories, could be either a relative path to a data folder, or a full path in aws s3, starting with `s3://`

  * two columns for naming the output, where output would be named as `score_{col1}_{col2}.tif`

    * a column named `type`, specifying whether each row is a `center` image whose prediction would be written out, or a `neighbor` image

  * Also an example of the table format . Here I use `dir_gs` and `dir_os` as directories to images, and `tile_col` and `tile_row` as the naming columns to keep the naming system consistent with `learner`

    | tile_col | tile_row | dir_gs                                            | dir_os                                            | type     |
    | -------- | -------- | ------------------------------------------------- | ------------------------------------------------- | -------- |
    | 320      | 560      | images/planet/fix/GS/tile539601_736815_736967.tif | images/planet/fix/OS/tile539601_737029_737118.tif | center   |
    | 321      | 560      | images/planet/fix/GS/tile539602_736815_736967.tif | images/planet/fix/OS/tile539602_737029_737118.tif | neighbor |

All catalogs are in folder  `s3://activemapper/data_DL` of aws s3.



## Run Semantic Segmentation

### Quick Run

#### Parameter Settings

The system uses a `config,yaml` file to set up parameters, which are defined as:

* Train_Validate

  * bucket

    ​	-- project's s3 bucket

  * prefix_out

    ​	-- directory relative to bucket where output would be saved to, including loss graphs, evaluation metrics, and prediction if specified

  * dir_data

    ​	-- directory to the folder saving catalog csvs, and/or raw images and labels

  * catalog

    ​	-- catalog name for train and validation with the csv extension, e.g. semantic_catalog_train_val.csv

  * img_path_cols

    ​    -- list of column names for image directories in provided catalog; the default is `['dir_gs', 'dir_os']`

  * label_path_col

    ​	-- the column name for label directories in provided catalog, e.g. `dir_label`

  * train_group

    ​	-- list of label groups for train

  * validate_group

    ​	-- list of label groups for validation

  * patch_size

    ​	-- size of target grid/labels. Default is 200.

  * buffer

    ​	-- buffer to add when getting chips, so the chips fed into models are in size of $patch\_size + 2\times buffer $

  * composite_buffer

    ​	-- buffer applied on tiles when creating composites. Default is 11.

  * bands

    ​	-- total number of bands or variables to the model, e.g. each of the GS and OS image have 4 bands, so the total bands is 8

  * class_number

    ​	-- number of categories for semantic segmentation, e.g. binary classification has a class_number of 2

  * transformation

    ​	-- list of data augmentation to apply

    ​	-- provided methods are

    ​			"vflip" -- vertical flip

    ​			"hflip" -- horizontal flip

    ​			" dflip" -- diagonal flip

    ​			"rotate" -- rotation

    ​			"resize" -- resize, the default resize rate range is [0.8, 1.2]

    ​			"shift_brightness" -- shift image brightness, the default shift gamma range is [0.2, 1.2]

  * rotation_degrees

    ​	-- range of angle in degrees for rotation, default is between -90° and 90°

  * brightness_shift_subsets

    ​	-- number of bands/channels on dataset for each brightness shift

  * model

    ​	-- provided models are: Unet, PSPnet, DeepLab3, DeepLab3plus, DeepLab3plus2, ExFuse, GCN

    ​	-- DeepLab3plus is the original model written in publication, while DeepLab3plus2 has a small modification from it by replacing bilinear interpolation on upsampling path to transposed convolution

    ​	-- to add other models, please add the model class named in lowercases in a `.py` file to the `models` folder, and corresponding import statement to `models/__init__.py`

  * initial_params

    ​	-- full directory to pretrained model parameters

    ​	-- leave it blank if there's no pretrained model parameters

  * train_batch

    ​	-- batch size for train; this is the minimal unit to optimize models

    ​	-- a general principle is to set the batch size divisible by size of the training datasets and as large as possible considering GPU's memory limit; this guarantees variation within batches and the least number of  chips left alone . For example,I would use batch of 30 for a dataset of 869 chips, so there are 29 batches, where the smallest has 28 chips, while others have 29.

  * validate_batch

    ​	-- batch size for validation

  * epoch

    ​	-- number of iterations

  * criterion

    ​	-- loss function with initial parameter specified

    ​	-- it could be any from the [pytorch loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) or [package defined functions](https://github.com/agroimpacts/deeplearner/tree/devel/aws/deeplearner/losses)

  * optimizer

    ​	-- algorithm to find direction to decrease loss

    ​	-- provided optimizers are: sgd, nesterov, adam, and amsgrad

    ​	-- to add other optimizers, please refer to [torch.optim](https://pytorch.org/docs/stable/optim.html), and edit in function `get_optimizer` or class `ModelCompiler` in `compiler.py`

  * learning_rate_init

    ​	-- start learning rate

  * learning_rate_policy

   -- learning rate policy that determines the scheduler and when to change learning rate. Optitons are: StepLR, MultiStepLR, ReduceLROnPlateau, PolynomialLR, CyclicLR

  <!-- * learning_rate_decay

    ​	-- rate applied to decrease learning rate, it works with the `learning_rate_decay_step` to determine the learning rate of an epoch

  * learning_rate_decay_step

    ​	-- this decides on how many epoch would a decay applied on the learning rate

    ​	-- it combined with`learning_rate_init` and `learning_rate_decay` to determine the learning rate of an epoch as $lr = lr0 \times (learning\_rate\_decay^{\lfloor epoch/learning\_rate\_decay\_step\rfloor}$

    ​	-- -- currently we use a step decay strategy; please refer to [coursera: learning rate decay](https://www.coursera.org/lecture/deep-neural-network/learning-rate-decay-hjgIA) for more learning rate decay strategy -->

  * momentum

    ​	-- momentum applied on optimizer

  * gpu_devices:

    ​	-- list of indices of gpus that the model would run on

  * resume:

     -- True or False, whether to resume from a checkpoint file

  * resume_epoch:

     -- The epoch number to resume from


* Predict

  * bucket

    ​	-- project's s3 bucket

  * prefix_out

    ​	-- directory relative to bucket where prediction would be written

  * dir_data

    ​	-- directory to the folder saving catalog csvs, and/or raw images and labels

  * catalog:

    ​	-- catalog name for prediction with the csv extension, e.g. semantic_catalog_predict_5.csv

  * img_path_cols

    ​    -- list of column names for image directories in provided catalog, e.g. `['dir_gs', 'dir_os']`

  * patch_size

    ​	-- size of target grid, this is the size that the predicted subset would be written on

  * buffer

    ​	-- buffer to add when extracting chips; thus, input to models are in size of $patch\_size +2\times buffer$

    ​	-- neighboring chips are overlapped in $2\times buffer$, where the buffer are cropped and omitted when writing tifs

  * composite_buffer

    ​	--  buffer applied on tiles when creating composites. Here the project use 179 with retiled images

  * bands

    -- total number of bands or variables to the model

  * class_number

    ​	-- number of categories for semantic segmentation, e.g. binary classification has a class_number of 2

  * pred_batch

    ​	-- batch size for prediction

  * average_neighbors

    ​	-- whether to average overlaps with neighboring tiles in prediction

  * shrink_pixels

    ​	-- pixel numbers to cut out on each side before averging neighbors

  * model

    ​	-- please choose from these models: Unet, PSPnet, DeepLab3, DeepLab3plus, DeepLab3plus2, ExFuse, GCN

    ​	-- this parameter would be ignored if train and prediction run in the same process

  * initial_params

    ​	-- full s3 directory to the pretrained model parameters

    ​	-- this parameter would be ignored if train and prediction run in the same process, but is required for independent prediction running

  * gpu_devices

    ​	-- list of gpus' indices that model would run on

  * checkpoint_upload:

     -- integer or list. number of checkpoint files or list of indices to upload to s3. e.g. 2 to save last 2 checkpoint files, or [10, 100] to save checkpoint file from epoch 10 and epoch 100

#### Run run_it.py

There are two required parameters of `run_it.py`: `--do-train` and `--do-prediction`. Both defaults are `False`. Here is an command line example that do both train and prediction:

```bash
python run_it.py --do-train --do-prediction
```

Tips:

* To reduce artifacts on grid edges after mosaicking chips on prediction , you can set the `patch_size` and `buffer` for prediction to be a larger value than that is used in train



### Run Step by Step

You can create an notebook file and import the package to run it step by step.

#### Load data

To begin, import packages in need, and set up a local working folder

* import packages

  ``````python
  from deeplearner import *
  from torch import nn
  import yaml
  ``````

* set up local working folder and result folder in s3

  * This is where temporary local files would be saved

  ``````python
  # local temporary folder
  dir_work = '~/some/directory/tmp'
  if not os.path.exists(dir_work):
      os.mkdir(dir_work)
  else:
      os.system("cd .. & rm -rf {}".format(dir_work))
      os.mkdir(dir_work)
  os.chdir(dir_work)

  # s3
  bucket = 'activemapper'
  prefix_out = 'some/directory/under/s3/bucket'
  ``````



Then,  start loading data. There are three types of dataset by their usage: `train`, `validate` and `prediction`.

Let's do train and validation data first. As always, the first step would be to set up parameters. Please refer to  [parameter settings](#parameter-settings) for any of the parameters' definition.

* Set up parameters for training dataset

  ``````python
  dir_data = 's3://activemapper/data_DL'
  fn_train_catalog = 'semantic_catalog_train_val.csv'
  img_path_cols = ['dir_gs', 'dir_os']
  label_path_col = 'dir_label'
  train_group = [2, 3, 4]
  buffer = 12
  composite_buffer = 11
  patch_size = 200
  train_batch = 32
  transformation = ['vflip', 'hflip', 'rotate', 'resize', 'shift_brightness']
  rotate_degree = [-90, 90]
  brightness_shift_subsets = [4, 4]
  ``````

* Load training dataset using `planetData` and slice them into batches using `DataLoader`.

  ```python
  # load catalog
  train_catalog = pd.read_csv(os.path.join(dir_data, fn_train_catalog))
  # load train dataset
  train_data = planetData(dir_data, train_catalog, patch_size, buffer,
                          composite_buffer, "train", imgPathCols=img_path_cols,
                          labelPathCol=label_path_col, labelGroup=train_group,
                          deRotate=rotate_degree, bShiftSubs = brightness_shift_subsets,
                          trans=transformation)
  train_dataloader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
  ```

* Load validation dataset. Most parameters are consistent with training ones, so you just need to specify label group and batch size for validation.

  ```python
  val_group = [3, 4]
  val_batch = 10

  validate_data = planetData(dir_data, train_catalog, patch_size, buffer,
                            composite_buffer, "validate", imgPathCols=img_path_cols,
                               labelPathCol=label_path_col, labelGroup=val_group)
  validate_dataloader = DataLoader(validate_data, batch_size=val_batch, shuffle=False)
  ```



#### Initiate model

The first step for running a model would be to initiate it. The model could be an existing one in the package, or any customized semantic segmentation model.

##### Initiate an existing model

* Choose one of models below

  * U-Net : "unet"
  * DeepLab v3: "deeplab3"
  * DeepLab v3+: "deeplab3plus" or " deeplab3plus2 "
  * ExFuse: "exfuse"
  * GCN: "gcn"
  * PSP-Net: "pspnet"

* Specify the number of input bands, class numbers, list of gpus indices and directory to your initial parameters. If there's no pretrained model parameters, leave it as `None`. Here is an example with DeepLab v3+:

  ```python
  img_bands = 8
  class_number = 2
  gpus = [0,1]
  init_params = None # put a full s3 directory if there's a trained model
  model = eval("deeplab3plus2".lower())(img_bands, class_number)
  ```

* Initiate a `ModelCompiler` with model parameters and gpus.

  ```python
  model = ModelCompiler(model, buffer, gpus, init_params)
  ```

##### Initiate a customized model

* For example, define a customized unet

  ```python
  class Conv3x3_bn_relu(nn.Module):
      def __init__(self, inch, outch, padding = 0, stride =1, dilation = 1, groups = 1, relu = True):
          super(Conv3x3_bn_relu, self).__init__()
          self.applyRelu = relu

          self.conv = nn.Sequential(nn.Conv2d(inch, outch, 3, padding = padding, stride = stride, dilation = dilation, groups = groups),
                                    nn.BatchNorm2d(outch))
          if self.applyRelu:
              self.relu = nn.ReLU(True)

      def forward(self, x):
          out = self.conv(x)
          if self.applyRelu:
              out = self.relu(out)
          return out

  class doubleConv(nn.Module):
      def __init__(self, inch, outch):
          super(doubleConv, self).__init__()
          self.conv1 = Conv3x3_bn_relu(inch, outch, padding = 1)
          self.conv2 = Conv3x3_bn_relu(outch, outch, padding = 1)

      def forward(self, x):
          x = self.conv1(x)
          x = self.conv2(x)
          return x

  class unet(nn.Module):
      def __init__(self, inch, classNum):
          super(unet, self).__init__()
          # downsample
          self.dlyr1 = doubleConv(inch, 64)
          self.ds = nn.MaxPool2d(2, stride=2)
          self.dlyr2 = doubleConv(64, 128)
          self.dlyr3 = doubleConv(128, 256)
          self.dlyr4 = doubleConv(256, 512)
          self.dlyr5 = doubleConv(512, 1024)
          self.dlyr6 = doubleConv(1024, 2048)

          # upsample
          self.us_init = nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1)
          self.ulyr_init = doubleConv(2048, 1024)
          self.us6 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
          self.ulyr6 = doubleConv(1024, 512)  # 512x32x32
          self.us7 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
          self.ulyr7 = doubleConv(512, 256)
          self.us8 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
          self.ulyr8 = doubleConv(256, 128)
          self.us9 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
          self.ulyr9 = doubleConv(128, 64)
          self.dimTrans = nn.Conv2d(64, classNum, 1)

      def forward(self, x):
          # downsample
          dlyr1 = self.dlyr1(x)
          ds1 = self.ds(dlyr1)
          dlyr2 = self.dlyr2(ds1)
          ds2 = self.ds(dlyr2)
          dlyr3 = self.dlyr3(ds2)
          ds3 = self.ds(dlyr3)
          dlyr4 = self.dlyr4(ds3)
          ds4 = self.ds(dlyr4)
          dlyr5 = self.dlyr5(ds4)
          ds_last = self.ds(dlyr5)
          dlyr_last = self.dlyr6(ds_last)

          # upsample
          us_init = self.us_init(dlyr_last)
          ulyr_init = self.ulyr_init(torch.cat([us_init, dlyr5], 1))
          us6 = self.us6(ulyr_init)
          merge6 = torch.cat([us6, dlyr4], 1)  # channel is the second dimension after batch operation
          ulyr6 = self.ulyr6(merge6)
          us7 = self.us7(ulyr6)
          merge7 = torch.cat([us7, dlyr3], 1)
          ulyr7 = self.ulyr7(merge7)
          us8 = self.us8(ulyr7)
          merge8 = torch.cat([us8, dlyr2], 1)
          ulyr8=self.ulyr8(merge8)
          us9 = self.us9(ulyr8)
          merge9 = torch.cat([us9, dlyr1], 1)
          ulyr9 = self.ulyr9(merge9)
          dimTrans = self.dimTrans(ulyr9)

          return dimTrans
  ```

* Then initiate the model using `ModelCompiler`

  ```python
  img_bands = 8
  class_numbers = 2
  gpus = [0,1]
  init_params = None

  model = unet(img_bands, class_number)
  model = ModelCompiler(model, buffer, gpus, init_params)
  ```



#### Train and validate model

* Set up following parameters:

  * `epoch`: number of iterations
  * `lr_init`: initial learning rate
  * `lr_policy`: policy for learning rate decay, options are "StepLR", "MultiStepLR", "ReduceLROnPlateau", "PolynomialLR", "CyclicLR"
  * `criterion`: loss function to calculate difference between model output and ground truth
  * `momentum`: momentum to apply on the loss function
  * `optimizer`: algorithm to find direction for reducing loss, currently provide "sgd", "nesterov",  "adam", "amsgrad". Others would be added in the future
  * `lr_params` include the appropriate arguments for the corresponding learning rate policy.

  * Default `lr_params` for different LR policies:
      * StepLR: {"step_size":3, "gamma":0.98}
      * MultiStepLR: {"milestones":[15, 25, 35, 50, 70, 90, 120, 150, 200], "gamma":0.5}
      * ReduceLROnPlateau: {"mode"='min', "factor":0.8, "patience":3, "threshold":0.0001, "threshold_mode":'rel', "min_lr":3e-6, "verbose":True}
      * PolynomialLR: {"max_decay_steps":100, "min_learning_rate":1e-5, "power":0.80}
      * CyclicLR: {"base_lr":3e-5, "max_lr":0.01, "step_size_up":1100, "mode":'triangular'}


  ```python
  epoch = 300
  lr_init = 0.01
  lr_policy = "StepLR"
  momentum = 0.95
  criterion = BalancedCrossEntropyLoss()
  optimizer = "nesterov"
  lr_params = {"step_size":3, "gamma":0.98}
  ```

* Use the `fit` method to train and validate the models.

  <!-- ```python
  model.fit(train_dataloader, validate_dataloader, epoch, optimizer, lr_init, (lr_decay, lr_decay_step), criterion, momentum)
  ``` -->
  ```python
  model.fit(train_dataloader, validate_dataloader, epoch, optimizer, lr_init, lr_policy, criterion, momentum, lr_params)
  ```



#### Evaluate model

Use the `evaluate` method to produce a csv of evaluation metrics on validation data in s3 output folder

```python
model.evaluate(validate_dataloader, bucket, prefix_out)
```



#### Save model

Use the `save` method to upload model parameters to s3. Use `save_checkpoint` method to upload checkpoint files to s3.

```python
model.save(bucket, prefix_out)
model.save_checkpoint(bucket, prefix_out, checkpoint_upload)
```

* checkpoint_upload:

   -- integer or list. number of checkpoint files or list of indices to upload to s3. e.g. 2 to save last 2 checkpoint files, or [10, 100] to save checkpoint file from epoch 10 and epoch 100


#### Prediction

* Define a function for loading prediction data on a single row in predict catalog.

  ```python
  def load_pred_data(dir_data, pred_patch_size, pred_buffer, pred_composite_buffer,
                     pred_batch, catalog, catalog_row, average_neighbors=False):
      def load_single_tile(catalog_ind = catalog_row):
          dataset = planetData(dir_data, catalog, pred_patch_size, pred_buffer,
                               pred_composite_buffer, "predict",
                               catalogIndex=catalog_ind, imgPathCols=img_path_cols)
          data_loader = DataLoader(dataset, batch_size=pred_batch, shuffle=False)
          meta = dataset.meta
          tile = dataset.tile
          return data_loader, meta, tile

      if average_neighbors == True:
          catalog["tile_col_row"] = catalog.apply(lambda x: "{}_{}".format(x['tile_col'], x['tile_row']), axis=1)
          tile_col = catalog.iloc[catalog_row].tile_col
          tile_row = catalog.iloc[catalog_row].tile_row
          row_dict = {
              "center": catalog_row,
              "top": catalog.query('tile_col=={} & tile_row=={}'.format(tile_col, tile_row - 1)).iloc[0].name \
                  if "{}_{}".format(tile_col, tile_row - 1) in list(catalog.tile_col_row) else None,
              "left" : catalog.query('tile_col=={} & tile_row=={}'.format(tile_col - 1, tile_row)).iloc[0].name \
                  if "{}_{}".format(tile_col - 1, tile_row) in list(catalog.tile_col_row) else None,
              "right" : catalog.query('tile_col=={} & tile_row=={}'.format(tile_col + 1, tile_row)).iloc[0].name \
                  if "{}_{}".format(tile_col + 1, tile_row) in list(catalog.tile_col_row) else None,
              "bottom": catalog.query('tile_col=={} & tile_row=={}'.format(tile_col, tile_row + 1)).iloc[0].name \
                  if "{}_{}".format(tile_col, tile_row + 1) in list(catalog.tile_col_row) else None,
              }
          dataset_dict = {k:load_single_tile(catalog_ind = row_dict[k]) if row_dict[k] is not None else None
                          for k in row_dict.keys()}
          return dataset_dict
      # direct crop edge pixels
      else:
          return load_single_tile()
  ```

* Set up parameters to load prediction data:

  * pred_patch_size: size of tile's subset to write into tifs; it is not necessarily the same as training chips

  * pred_buffer:  buffer to add when extracting chips, where neighboring chips are overlapped in $2\times buffer$

  * pred_composite_buffer: number of padded pixels added to each side when creating the input image; the predicted scores would be in size of $input\_image\_size - 2\times pred\_composite\_buffer$

  * pred_batch: batch size of prediction dataset

  * shrink_pixels: pixel numbers to cut out on each side before averging neighbors

  * average_neighbors: whether to average overlaps with neighboring tiles

    ```python
    pred_patch_size = 250
    pred_buffer = 179
    pred_composite_buffer = 179
    pred_batch = 2
    shrink_pixels = 54
    average_neighbors = True
    ```

* Read each row marked as `center` in the prediction catalog, build dataset, and predict

  ```python
  fn_pred_catalog = "semantic_catalog_predict_retiled_5.csv"
  pred_catalog = pd.read_csv(os.path.join(dir_data, fn_pred_catalog))

  # only predict on images marked as 'center' in the catalog
  inds = pred_catalog.query("type == 'center'").index.values
  for i in inds:
      pred_dataloader = load_pred_data(dir_data, pred_patch_size, pred_buffer,
                                       pred_composite_buffer,pred_batch, pred_catalog,
                                       i, average_neighbors = average_neighbors)
      model.predict(pred_dataloader, bucket, prefix_out, pred_buffer,
                    averageNeighbors=average_neighbors, shrinkBuffer = shrink_pixels)
  ```
