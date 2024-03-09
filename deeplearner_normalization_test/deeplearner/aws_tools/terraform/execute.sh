#!/bin/bash
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# input variables
IID=$1
PYTHONARGS=$2
S3LOGDIR=$3

# clone repo
git clone -b devel/aws_terraform https://github.com/agroimpacts/deeplearner.git
cd deeplearner

# redirect bash output to stdout.log
LOCALLOGPATH=log-$IID
LOGNAME=stdout-$(date +"%Y-%m-%dT%T").log
[ ! -d "$LOCALLOGPATH" ] && mkdir $LOCALLOGPATH
exec &>> $LOCALLOGPATH/$LOGNAME

# activate conda and run python
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_latest_p36
python run_it.py $PYTHONARGS

# at the end
## backup log file to s3
aws s3 sync $LOCALLOGPATH $S3LOGDIR/$LOCALLOGPATH
