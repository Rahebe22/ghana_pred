#!/usr/bin/env bash
## create a spot instance from AMI

source ~/.bashrc
# Get input parameters
AMIID=$1
if [ -z "$AMIID" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <bid_price>"
    exit 1
fi
ITYPE=$2
if [ -z "$ITYPE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <bid_price>"
    exit 1
fi
SECGROUPID=$3
if [ -z "$SECGROUPID" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <bid_price>"
    exit 1
fi
NEWINAME=$4
if [ -z "$NEWINAME" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <bid_price>"
    exit 1
fi
SPOTTYPE=$5
if [ -z "$SPOTTYPE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <bid_price>"
    exit 1
fi
BIDPRICE=$6
if [ -z "$BIDPRICE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <bid_price>"
    exit 1
fi



## Set up new instance
echo "Setting up new spot instance named $NEWINAME from AMI $AMIID in $ZONE on a bid_price of $BIDPRICE"

aws ec2 run-instances \
	--image-id $AMIID \
	--count 1 \
	--instance-type $ITYPE \
	--iam-instance-profile 'Name="activemapper_planet_readwriteS3"' \
	--key-name boka-key-pair \
	--security-group-ids $SECGROUPID \
	--tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value='$NEWINAME'}]' \
	--instance-market-options 'MarketType=spot,
        SpotOptions={MaxPrice='$BIDPRICE',
		SpotInstanceType='$SPOTTYPE',
		InstanceInterruptionBehavior=terminate}'

NEWIID=$(aws ec2 describe-instances \
	--filters 'Name=tag:Name,Values='"$NEWINAME"'' \
	--output text --query 'Reservations[*].Instances[*].InstanceId')


echo $NEWIID
