resource "aws_key_pair" "gpu" {
  key_name   = "toolcall-serving-bench-gpu"
  public_key = file(pathexpand("~/.ssh/id_ed25519.pub"))
}
