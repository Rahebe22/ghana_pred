import click
import yaml
import urllib.parse as urlparse

from deeplearner import *

def run_segmentation(configPath, doTrain, doPrediction):

    # temporary working folder
    dir_work = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(dir_work):
        os.mkdir(dir_work)
    else:
        os.system("cd .. & rm -rf {}".format(dir_work))
        os.mkdir(dir_work)
    # config file
    if configPath.startswith("s3"):
        parsed = urlparse.urlparse(configPath)
        params=yaml.load(boto3.resource('s3').Bucket(parsed.netloc).Object(parsed.path[1:]).get()['Body'].read())
    else:
        with open(configPath, "r") as config:
            params = yaml.safe_load(config)

    if doTrain:

        # parameters
        params_train = params['Train_Validate']

        ## Parameters concerning data in s3
        prefix_out = params_train['prefix_out']
        bucket = params_train['bucket']
        class_number = params_train['class_number']
        img_bands = params_train['bands']
        buffer = params_train['buffer']

        ## Parameters concerning model
        os.chdir(dir_work)
        model = eval(params_train['model'].lower())(img_bands, class_number)
        init_params = params_train['initial_params']
        freeze_params = params_train['freeze_params']
        freeze_params = freeze_params if isinstance(freeze_params, list) or freeze_params==None else eval(freeze_params)
        ## Parameters concerning train and validate
        epoch = params_train['epoch']
        optimizer = params_train['optimizer']
        lr_init = params_train['learning_rate_init']
        lr_decay = (params_train['learning_rate_decay'], params_train['learning_rate_decay_step'])
        momentum = params_train['momentum']
        criterion = eval(params_train['criterion'])
        gpus = params_train['gpu_devices']
        resume = params_train['resume']
        resume_epoch = params_train['resume_epoch']
        checkpoint_upload = params_train['checkpoint_upload']

        # Load data
        print('Loading training dataset')
        train_dataset = load_dataset(params_train, "train")

        print('Loading validation dataset')
        validate_dataset = load_dataset(params_train, "validate")

        # Train and validate
        print('Compiling and training model')
        model = ModelCompiler(model, buffer, gpus, init_params, freeze_params)
        model.fit(train_dataset, validate_dataset, epoch, optimizer, lr_init, lr_decay, criterion,
                  momentum, resume, resume_epoch, bucket, prefix_out)

        ## Evaluate and save model
        print('Evaluating model')
        model.evaluate(validate_dataset, bucket, prefix_out)
        model.save(bucket, prefix_out)
        model.save_checkpoint(bucket, prefix_out, checkpoint_upload)

    ## Prediction
    if doPrediction:

        params_pred = params['Predict']
        # Get indices for prediction
        inds = pd.read_csv(os.path.join(params_pred['dir_data'], params_pred['catalog'])) \
            .query("type == 'center'") \
            .index \
            .values
        prefix_out = params_pred['prefix_out']
        buffer_pred = params_pred['buffer']
        shrink_pixels = params_pred['shrink_pixels'] if params_pred['shrink_pixels'] else 0
        average_neighbors = params_pred['average_neighbors']

        # inherit model from trian
        if doTrain:
            pass
        # else initiate model from saved parameters
        else:
            model = eval(params_pred['model'].lower())(params_pred['bands'], params_pred['class_number'])
            init_params = params_pred['initial_params']
            gpus = params_pred['gpu_devices']

            bucket = params_pred['bucket']
            model = ModelCompiler(model, buffer_pred, gpus, init_params)

        # Load each tile and predict
        for i in inds:
            attempt = 0
            while attempt < 5:
                try:
                    pred_dataset = load_dataset(params_pred, "predict", i)
                    model.predict(pred_dataset, bucket, prefix_out, buffer_pred, averageNeighbors=average_neighbors, shrinkBuffer = shrink_pixels)
                    break
                except:
                    attempt += 1

    print("Backing up working folder to s3")
    backup_folder_to_s3(dir_work, bucket, prefix_out)
    # delete tmp folder
    os.system("cd .. & rm -rf {}".format(dir_work))


@click.command()
@click.option('--config', default='./config.yaml',
              help='Directory of the config to use')
@click.option('--do-train', is_flag=True, help='Do train model')
@click.option('--do-prediction', is_flag=True, help='Do prediction on specified data')
def main(config, do_train, do_prediction):
    run_segmentation(config, do_train, do_prediction)


if __name__ =='__main__':
    main()
