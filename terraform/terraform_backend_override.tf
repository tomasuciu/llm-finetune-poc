terraform {
  backend "s3" {
    bucket = "tfstate-slurm-k8s-d3521d766aeb184f748caacb0196c126"
    key    = "slurm-k8s.tfstate"

    endpoints = {
      s3 = "https://storage.eu-north1.nebius.cloud:443"
    }
    region = "eu-north1"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}
