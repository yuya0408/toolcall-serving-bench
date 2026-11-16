terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.23.0"
    }
  }

  # the bucket comes from infra/backend.hcl (gitignored; copy backend.hcl.example):
  #   terraform -chdir=infra init -backend-config=backend.hcl
  backend "s3" {
    key          = "cloud-layer/terraform.tfstate"
    region       = "ap-northeast-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = "ap-northeast-1"
}
