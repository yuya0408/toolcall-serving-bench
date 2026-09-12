resource "aws_security_group" "gpu_ssh" {
  name        = "toolcall-serving-bench-gpu-ssh"
  description = "Allow SSH only from the operator's IP; no inference ports exposed"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
