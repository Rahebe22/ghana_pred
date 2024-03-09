# create a Linux instance in AWS
# execute bash script to set up environment and run codes
provider "aws" {
        version    = "~> 2.0"
        region     = var.region
}

# Request a spot instance at $0.03
resource "aws_spot_instance_request" "dltest" {
  ami           = var.ami
  spot_price    = var.bid_price
  instance_type = var.instance_type
  key_name = var.key_name
  security_groups = var.security_group
  
  wait_for_fulfillment = true
  spot_type = var.spot_type
  iam_instance_profile = var.iam_role  
  tags = {
    Name = var.instance_name
  }

  # ssh to the instance and execute bash scripts
  connection {
    type        = "ssh"
    user        = var.ssh_user
    private_key = file(var.key_path)
    host        = self.public_ip    
  }

  provisioner "local-exec" {
    command = "aws ec2 create-tags --resources ${self.spot_instance_id} --tags Key=Name,Value=${var.instance_name}"
  }

  # Copy execute.sh to remote machine
  provisioner "file" {
    source      = "./execute.sh"
    destination = "~/execute.sh"

  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x ~/execute.sh",
      "~/execute.sh ${self.spot_instance_id} '--config s3://activemapper/DL/configs/config_test.yaml --do-train' ${var.s3_log_path}"]
  }
  provisioner "local-exec" {
    command = "aws ec2 cancel-spot-instance-requests --spot-instance-request-ids ${self.id}"
  }
  provisioner "local-exec" {
    command = "aws ec2 terminate-instances --instance-ids ${self.spot_instance_id}"
  }
}
