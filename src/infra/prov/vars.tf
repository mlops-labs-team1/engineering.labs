# Configurations for Google Cloud Provider
variable "gcp_region" {
  description = "GCP Region to use"
  default     = "us-east1"
}

variable "gcp_zone" {
  description = "GCP Zone to use"
  default     = "us-east1-a"
}

variable "mlflow_bucket_name" {
  default = "mlflow-data-store"
}


variable "ip_cidr_range" {
  description = "IP CIDR range for the subnetwork"
  default     = "10.20.30.0/24"
}

variable "vm_image" {
  description = "SO Image to be used"
  default     = "ubuntu-1804-bionic-v20201014"
}

variable "ansible_ssh" {
  type        = map(string)
  description = "SSH public key file used by ansible to configure the server"
  default = {
    user     = "ansible"
    key_file = "ansible-key.pub"
  }
}

variable "mlflow_db_name" {
  description = "MLFlow DB name"
  default     = "mlflow_db"
}


# Mandatory variables
variable "mlflow_db_user" {
  description = "MLFlow user to connect to DB"
}

variable "mlflow_db_pass" {
  description = "MLFlow user pass to connect to DB"
}

variable "gcp_project" {
  description = "GCP Project Id"
}

