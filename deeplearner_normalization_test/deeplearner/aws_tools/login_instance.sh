#!/usr/bin/env bash

# Get input parameters
INAME=$1
if [ -z "$INAME" ]; then
    echo "`date`: Usage: $0 <instance_name> <user_name>"
    exit 1
fi

USER=$2
if [ -z "$USER" ]; then
    echo "`date`: Usage: $0 <instance_name> <user_name>"
    exit 1
fi

# get public dns ip
IIP=$(aws ec2 describe-instances \
	--filters 'Name=tag:Name,Values='"$INAME"'' \
	--output text \
	--query 'Reservations[*].Instances[*].PublicDnsName')

# log in
ssh $USER@$IIP