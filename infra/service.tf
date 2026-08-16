resource "aws_ecs_express_gateway_service" "smoke_test" {
  service_name             = "smoke-test"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  infrastructure_role_arn  = aws_iam_role.ecs_infrastructure.arn
  health_check_path        = "/"

  primary_container {
    image          = "${aws_ecr_repository.smoke_test.repository_url}:latest"
    container_port = 8080
  }

  scaling_target {
    min_task_count            = 1
    max_task_count            = 1
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = 60
  }

  depends_on = [
    aws_iam_role_policy_attachment.ecs_task_execution,
    aws_iam_role_policy_attachment.ecs_infrastructure,
  ]
}

output "application_url" {
  value = aws_ecs_express_gateway_service.smoke_test.ingress_paths
}
