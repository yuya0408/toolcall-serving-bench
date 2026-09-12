variable "ssh_allowed_cidr" {
  description = "SSH接続を許可するグローバルIPのCIDR。回線側でIPが変わったらここを更新してreapplyする"
  type        = string
  default     = "110.135.196.164/32"
}

variable "auto_shutdown_minutes" {
  description = "インスタンス起動(boot)からの自動シャットダウンまでの分数。止め忘れ・ハング時のコスト超過バックストップ"
  type        = number
  default     = 360
}
