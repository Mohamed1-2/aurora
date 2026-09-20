output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "aks_name" {
  value = azurerm_kubernetes_cluster.aks.name
}

output "aks_connect" {
  value = format("az aks get-credentials --resource-group %s --name %s --overwrite-existing", azurerm_resource_group.main.name, azurerm_kubernetes_cluster.aks.name)
}

output "mlflow_artifact_store" {
  value = format("az://mlflow/%s", azurerm_storage_container.mlflow.name)
}