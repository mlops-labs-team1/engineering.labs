resource "google_compute_network" "englab_vpc" {
  name                    = "englab-vpc"
  description             = "EngLabs default VPC"
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"
}

resource "google_compute_subnetwork" "englab_subnet" {
  name          = "englab-subnet"
  ip_cidr_range = var.ip_cidr_range
  network       = google_compute_network.englab_vpc.id
}

resource "google_compute_firewall" "internal_traffic" {
  name    = "englab-internal-traffic"
  network = google_compute_network.englab_vpc.name

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  source_ranges = [var.ip_cidr_range]
}

resource "google_compute_firewall" "allow_web" {
  name    = "englab-allow-web"
  network = google_compute_network.englab_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  target_tags = ["web"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "englab-allow-ssh"
  network = google_compute_network.englab_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  target_tags = ["ssh"]
}

resource "google_compute_firewall" "tracking_node" {
  name    = "englab-tracking-node"
  network = google_compute_network.englab_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }

  target_tags = ["tracking"]
}

/*
 * We need a static IP allocation for Tracking Node so we can build a link between it and DB instance.
 * Doing that for all nodes so we will need to generate Ansible inventory only once and after resources
 * creation.
 */
resource "google_compute_address" "static_ip" {
  for_each = toset(["training", "tracking"])
  name     = "${each.key}-static-ip"
}