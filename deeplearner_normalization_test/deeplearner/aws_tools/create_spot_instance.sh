#!/usr/bin/env sh
## create a spot instance from AMI
# source ~/.bashrc
# Get input parameters
AMIID=$1
if [ -z "$AMIID" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
fi
ITYPE=$2
if [ -z "$ITYPE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
fi
SECGROUPID=$3
if [ -z "$SECGROUPID" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
fi
KEYPAIR=$4
if [ -z "$KEYPAIR" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
fi
IAMROLE=$5
if [ -z "$IAMROLE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
fi
NEWINAME=$6
if [ -z "$NEWINAME" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
fi
SPOTTYPE=$7
if [ -z "$SPOTTYPE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type>"
    exit 1
elif [[ "$SPOTTYPE" == "one-time" ]]; then
    INTERRUPTION='terminate'
elif [[ "$SPOTTYPE" == "persistent" ]]; then
    INTERRUPTION='stop'
else
    echo "spot type not applicable"
    exit 1
fi

# get bid price
# check OS first: if mac, need to brew install coreutils to get gdate
OS=$(uname -s)
if [[ "$OS" == "Linux" ]]; then
  alias ddate='date'
elif [[ "$OS" == "Darwin" ]]; then
  alias ddate='gdate'
else
  echo "Not linux or mac"
  exit 1
fi

START_TIME=$(ddate --date="3 days ago" +"%Y-%m-%dT%T")
END_TIME=$(ddate +"%Y-%m-%dT%T")

read -r -d '' PRICES << EOF
    $(aws ec2 describe-spot-price-history --instance-types $ITYPE \
		--product-description Linux/UNIX \
		--start-time $START_TIME \
		--end-time $END_TIME)
EOF

## find highest price zone and get the max spot price in that zone
ZONE=$(echo $PRICES |\
  jq '[.SpotPriceHistory[]] | sort_by(.AvailabilityZone | explode | map(-.)) |
  max_by(.SpotPrice | tonumber)|.AvailabilityZone')

MAX_SPOT_PRICE=$(echo $PRICES |\
	jq '[.SpotPriceHistory[] | select(.AvailabilityZone == '"$ZONE"')] |
	max_by(.SpotPrice | tonumber) |.SpotPrice |tonumber')

## get bid price by adding an overflow
OVERFLOW=0.05
BID_PRICE=$(echo | awk -v a=$MAX_SPOT_PRICE -v b=$OVERFLOW '{print a+b}')

## get subnetId of lowest price zone
SUBNETID=$(aws ec2 describe-subnets \
		--filter 'Name=availability-zone,Values='$ZONE'' \
		           'Name=vpc-id,Values=vpc-e48b1a9d' \
		--output text \
		--query 'Subnets[*].SubnetId')

## Set up new instance
echo "Setting up new spot instance named $NEWINAME from AMI $AMIID on a bid_price of $BID_PRICE"

aws ec2 run-instances \
	--image-id $AMIID \
	--count 1 \
	--instance-type $ITYPE \
	--iam-instance-profile 'Name="'$IAMROLE'"' \
	--key-name $KEYPAIR \
	--security-group-ids $SECGROUPID \
	--tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value='$NEWINAME'}]' \
	--instance-market-options 'MarketType=spot,
    SpotOptions={MaxPrice='$BID_PRICE',
		SpotInstanceType='$SPOTTYPE',
		InstanceInterruptionBehavior='$INTERRUPTION'}'
NEWIID=$(aws ec2 describe-instances \
	--filters 'Name=tag:Name,Values='"$NEWINAME"'' \
	--output text --query 'Reservations[*].Instances[*].InstanceId')


echo $NEWIID


