/*
 *  Service account assigned to instances that uses our private docker registry and
 *  model repository
*/
resource "google_service_account" "mlflow_data_account" {
  account_id   = "mlflow-data"
  display_name = "MlFlow Data Service Account"
}

resource "google_project_iam_member" "iam_registry_writer" {
  role   = "roles/artifactregistry.writer"
  member = "serviceAccount:${google_service_account.mlflow_data_account.email}"
}

resource "google_project_iam_member" "iam_storage_writer" {
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlflow_data_account.email}"
}

resource "google_project_iam_member" "iam_storage_admin" {
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.mlflow_data_account.email}"
}

resource "google_project_iam_member" "iam_run_admin" {
  role   = "roles/run.admin"
  member = "serviceAccount:${google_service_account.mlflow_data_account.email}"
}

# In order to deploy in Cloud Run
resource "google_project_iam_member" "iam_act_as" {
  role   = "roles/iam.serviceAccountUser"
  member = "serviceAccount:${google_service_account.mlflow_data_account.email}"
}