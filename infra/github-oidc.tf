data "tls_certificate" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github_actions.certificates[0].sha1_fingerprint]
}

data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # yuya0408/toolcall-serving-bench リポジトリからのワークフロー実行のみ信頼する。
    # GitHubのOIDCトークンのsubクレームは owner/repo名だけでなく不変ID(@数字)を含む形式
    # （repo:OWNER@OWNER_ID/REPO@REPO_ID:ref:...）で発行されるため、名前だけの条件では一致しない。
    # 実際にCloudTrailで観測したsubクレーム、およびgh api users/yuya0408・repos/yuya0408/toolcall-serving-bench
    # のidフィールドと突き合わせて確認済み（owner_id=240127008, repo_id=1335809260）。
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:yuya0408@240127008/toolcall-serving-bench@1335809260:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "github-actions-tool-call-bench"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json
}

# Scope: CI only builds/pushes the smoke-test image and redeploys the ECS
# service. It does not run terraform apply/destroy — infra changes stay a
# manual, local operation — so no IAM/OIDC-provider/state-bucket permissions
# are granted here. That keeps a malicious or buggy PR from being able to
# escalate this role's own permissions through CI.
data "aws_iam_policy_document" "github_actions_permissions" {
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # AWS requires resources = "*" for this action
  }

  statement {
    sid = "EcrPushSmokeTest"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
    ]
    resources = [aws_ecr_repository.smoke_test.arn]
  }

  statement {
    sid = "EcsRedeploySmokeTest"
    actions = [
      "ecs:ListClusters",
      "ecs:ListServices",
      "ecs:DescribeServices",
      "ecs:UpdateService",
    ]
    # aws_ecs_express_gateway_service does not expose a plain resource ARN we
    # can pin here (it provisions its own cluster/ALB), so the deploy workflow
    # discovers the cluster/service at runtime (ListClusters/ListServices)
    # instead of a hardcoded name, and this is scoped by action set rather
    # than resource. Tighten to a specific service ARN once confirmed (e.g.
    # via CloudTrail after the first real workflow run).
    resources = ["*"]
  }
}

resource "aws_iam_policy" "github_actions" {
  name   = "github-actions-tool-call-bench-deploy"
  policy = data.aws_iam_policy_document.github_actions_permissions.json
}

resource "aws_iam_role_policy_attachment" "github_actions_deploy" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.github_actions.arn
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}
