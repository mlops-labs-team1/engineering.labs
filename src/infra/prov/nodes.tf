resource "google_compute_instance" "training_node" {
  name                      = "training-node"
  machine_type              = "c2-standard-8"
  tags                      = ["ssh", "training", "tf-created"]
  zone                      = var.gcp_zone
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "ubuntu-2004-focal-v20210105"
      size  = 100
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.englab_subnet.name

    access_config {
      nat_ip = google_compute_address.static_ip["training"].address
    }
  }

  metadata = {
    "ssh-keys" = "${var.ansible_ssh.user}:${file(var.ansible_ssh.key_file)}"
  }

  /* So now we can seamless use registry and storage APIs from this instance
   * Check iam.tf to loock for its definition
   */
  service_account {
    email  = google_service_account.mlflow_data_account.email
    scopes = ["cloud-platform"]
  }
}

resource "google_compute_instance" "tracking_node" {
  name                      = "tracking-node"
  machine_type              = "g1-small"
  tags                      = ["ssh", "tracking", "tf-created"]
  zone                      = "southamerica-east1-a"
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "ubuntu-2004-focal-v20210105"
      size  = 15
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.englab_subnet.name

    access_config {
      # Need a static IP to configure in DB instance
      nat_ip = google_compute_address.static_ip["tracking"].address
    }
  }

  metadata = {
    "ssh-keys" = "${var.ansible_ssh.user}:${file(var.ansible_ssh.key_file)}"
  }

  /* So now we can seamless use registry and storage APIs from this instance
   * Check iam.tf to loock for its definition
   */
  service_account {
    email  = google_service_account.mlflow_data_account.email
    scopes = ["cloud-platform"]
  }
}