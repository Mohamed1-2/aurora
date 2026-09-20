resource "azurerm_resource_group" "main" {
  name     = "aurora-rg"
  location = var.location
}

# --- Container registry for image artifacts ---
resource "azurerm_container_registry" "acr" {
  name                = "auroraacr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.acr_sku
  admin_enabled       = false
}

# --- Key Vault: holds model / provider secrets surfaced via CSI or External Secrets ---
resource "azurerm_key_vault" "kv" {
  name                       = "aurorakv"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  tenant_id                  = var.tenant_id
  sku_name                   = "standard"
  enable_rbac_authorization  = true
  purge_protection_enforced  = false
}

resource "azurerm_key_vault_secret" "ai_endpoint" {
  name         = "aurora-gateway-azure-openai-endpoint"
  value        = "https://aurora-openai.openai.azure.com/"
  key_vault_id = azurerm_key_vault.kv.id
}

# --- Storage: reproducible artifact store backing the MLflow registry ---
resource "azurerm_storage_account" "mlflow" {
  name                     = "auroramlflowstore"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "mlflow" {
  name                  = "mlflow"
  storage_account_name  = azurerm_storage_account.mlflow.name
  container_access_type = "private"
}

# --- AKS: the runtime home for the AURORA micro-service fleet ---
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "aurora-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "aurora"

  default_node_pool {
    name       = "system"
    node_count = var.node_count
    vm_size    = var.vm_size
  }

  identity {
    type = "SystemAssigned"
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.law.id
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "calico"
  }
}

resource "azurerm_log_analytics_workspace" "law" {
  name                = "aurora-law"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
}

data "azurerm_subscription" "current" {}   # pragma: allowlist secret