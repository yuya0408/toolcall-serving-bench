resource "aws_ecr_repository" "smoke_test" {
  name                 = "smoke-test"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

output "ecr_repository_url" {
  value = aws_ecr_repository.smoke_test.repository_url
}
