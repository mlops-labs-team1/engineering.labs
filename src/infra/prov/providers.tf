terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.52.0"
    }

    random = {
      source  = "hashicorp/random"
      version = "3.0.1"
    }
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

provider "google-beta" {
  project = var.gcp_project
  region  = var.gcp_region
}