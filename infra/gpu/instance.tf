locals {
  instance_type = "g6.xlarge"
}

resource "aws_instance" "gpu" {
  ami           = data.aws_ssm_parameter.dlami.value
  instance_type = local.instance_type
  # g6.xlargeを提供しているAZのサブネットのうち最初のものを使う(data.tf)。
  # 提供はしていても在庫が無いとInsufficientInstanceCapacityで失敗するので、
  # その場合はindexを変えて再apply
  subnet_id                   = element(data.aws_subnets.default.ids, 0)
  vpc_security_group_ids      = [aws_security_group.gpu_ssh.id]
  key_name                    = aws_key_pair.gpu.key_name
  associate_public_ip_address = true

  instance_initiated_shutdown_behavior = "terminate"

  root_block_device {
    volume_size           = 200
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = <<-EOF
    #!/bin/bash
    shutdown -h +${var.auto_shutdown_minutes}
  EOF

  tags = {
    Name    = "toolcall-serving-bench-gpu"
    Project = "toolcall-serving-bench"
  }
}

output "gpu_public_ip" {
  value = aws_instance.gpu.public_ip
}

output "gpu_ssh_command" {
  value = "ssh -i ~/.ssh/id_ed25519 ubuntu@${aws_instance.gpu.public_ip}"
}
