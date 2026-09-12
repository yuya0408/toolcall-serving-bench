data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# PyTorchを含まないベース版（ドライバ+CUDA Toolkitのみ）。vLLMが自分でPyTorchを
# インストールするため、あらかじめ焼き込まれていると競合しやすい
data "aws_ssm_parameter" "dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}
