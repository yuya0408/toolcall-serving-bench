variable "ssh_allowed_cidr" {
  description = "SSH接続を許可するグローバルIPのCIDR（例：203.0.113.10/32）。既定値は置かず、applyのたびに今の回線のIPを渡す（runbookの0）"
  type        = string
}

variable "auto_shutdown_minutes" {
  description = "インスタンス起動(boot)からの自動シャットダウンまでの分数。止め忘れ・ハング時のコスト超過バックストップ"
  type        = number
  default     = 360
}
