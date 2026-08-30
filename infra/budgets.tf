# __generated__ by Terraform
# Please review these resources and move them into your main configuration files.

# __generated__ by Terraform
resource "aws_budgets_budget" "monthly" {
  account_id        = "344261432226"
  billing_view_arn  = null
  budget_type       = "COST"
  limit_amount      = "60.0"
  limit_unit        = "USD"
  name              = "tool-call-bench"
  tags              = {}
  tags_all          = {}
  time_period_end   = "2087-06-15_00:00"
  time_period_start = "2026-08-01_00:00"
  time_unit         = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["hirayuu0408@gmail.com"]
    subscriber_sns_topic_arns  = []
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["hirayuu0408@gmail.com"]
    subscriber_sns_topic_arns  = []
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["hirayuu0408@gmail.com"]
    subscriber_sns_topic_arns  = []
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
  }
}
