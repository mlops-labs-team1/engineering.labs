
resource "random_id" "db_suffix" {
  byte_length = 2
}

resource "google_sql_database_instance" "englab_db_instance" {

  /*  
   * It takes aprox. 1 week to free names from deleted db instances. 
   * Random Id provides an easy way for recreating your db instance.
   */
  name             = "englab-${random_id.db_suffix.hex}"
  database_version = "POSTGRES_12"

  /*  CAREFUL: Just uncomment line below if you know what you're doing!
   *  "deletion_protection" providers security for your even if you destroy
   *  or delete your data. If set to 'true' (default), Terraform won't delete
   *  your instance during a destroy command. You must set "delete_protecttion"
   *  to false before trying to destroy.
  */
  # deletion_protection = false

  settings {
    # Only custom or shared-core
    tier              = "db-g1-small"
    availability_type = "ZONAL"

    backup_configuration {
      enabled    = true
      start_time = "04:30"
    }

    ip_configuration {
      authorized_networks {
        name  = google_compute_instance.tracking_node.name
        value = google_compute_instance.tracking_node.network_interface.0.access_config.0.nat_ip
      }
      authorized_networks {
        name  = google_compute_instance.training_node.name
        value = google_compute_instance.training_node.network_interface.0.access_config.0.nat_ip
      }
    }
  }
}
resource "google_sql_user" "users" {
  name     = var.mlflow_db_user
  instance = google_sql_database_instance.englab_db_instance.name
  password = var.mlflow_db_pass
}
resource "google_sql_database" "database" {
  name     = var.mlflow_db_name
  instance = google_sql_database_instance.englab_db_instance.name
}

resource "google_storage_bucket" "mlflow_bucket_name" {
  name                        = var.mlflow_bucket_name
  location                    = var.gcp_region
  uniform_bucket_level_access = true
}