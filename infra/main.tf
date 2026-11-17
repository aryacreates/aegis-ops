terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "asia-south1"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_service_account" "aegis_sa" {
  account_id   = "aegis-ops-runner"
  display_name = "AegisOps ADK Cloud Run Service Account"
}

resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.aegis_sa.email}"
}

resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "aegis-ops"
  format        = "DOCKER"
}

resource "google_cloud_run_v2_service" "aegis_api" {
  name     = "aegis-ops-platform"
  location = var.region

  template {
    service_account = google_service_account.aegis_sa.email
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/aegis-ops/aegis-ops:latest"
      ports { container_port = 8080 }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
    }
  }
}