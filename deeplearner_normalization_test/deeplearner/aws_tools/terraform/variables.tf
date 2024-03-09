# variables.tf
variable "region" {
   type = "string"
   description = "AWS region"
   default = "us-east-1"
}

variable "instance_name" {
   type = "string"
   description = "Name of the EC2 Instance"
   default = "DLTest"
}

variable "instance_type" {
   type = "string"
   description = "Instance type to launch"
   default = "g4dn.2xlarge"
}

variable "bid_price" {
   type = "string"
   description = "Bid price, change depending on instance type"
   default = "0.4"
}

variable "spot_type" {
   type = "string"
   description = "Type of the spot instance, either persistent or one-time"
   default = "one-time"
}

variable "ami" {
   type = "string"
   description = "AMI to use for the EC2 instance"
   default = "ami-09f9fb464c46bb979"
}

variable "iam_role"{
   type = "string"
   description = "IAM role for the EC2 instance"
   default = "activemapper_planet_readwriteS3"
}

variable "s3_log_path"{
   type = "string"
   description = "The path in s3 to save log file"
   default = "s3://activemapper/DL/logs"
}

variable "ssh_user" {
   type = "string"
   description = "User for ssh log in"
   default = "terraform"
}

variable "key_name" {
   type = "string"
   description = "Name of the private key to create the instance"
   default = "boka-key-pair"
}

variable "key_path" {
   type = "string"
   description = "The path to a private key for SSH access"
   default = "~/.ssh/id_rsa"
}

variable "security_group" {
   type = "list"
   description = "The security group to use for the EC2 instance"
   default = ["jupyterhub"]
}

# end of variables.tf
