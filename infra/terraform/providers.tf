terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
  backend "azurerm" {
    resource_group_name  = "aurora-rg"
    storage_account_name = "auroratfstate"
    container_name       = "tfstate"
    key                  = "aurora/terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}

variable "subscription_id" { type = string }
variable "tenant_id"       { type = string }
variable "location"        { type = string, default = "eastus" }
variable "environment"     { type = string, default = "prod" }

variable "acr_sku"    { type = string, default = "Basic" }
variable "node_count" { type = number, default = 2 }
variable "vm_size"    { type = string, default = "Standard_D2s_v3" }