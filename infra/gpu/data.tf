data "aws_vpc" "default" {
  default = true
}

# g6.xlargeを提供しているAZだけに絞る。東京では1dが未提供で、
# 絞らずに先頭のサブネットを使うとRunInstancesがUnsupportedで失敗した
data "aws_ec2_instance_type_offerings" "gpu" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [local.instance_type]
  }
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.gpu.locations
  }
}

# PyTorchを含まないベース版（ドライバ+CUDA Toolkitのみ）。vLLMが自分でPyTorchを
# インストールするため、あらかじめ焼き込まれていると競合しやすい
data "aws_ssm_parameter" "dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}
