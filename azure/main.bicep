// =========================================================================
// Telecom Customer Churn — Azure Container Apps infrastructure.
//
// Provisions, in a single resource group:
//   * Log Analytics workspace (required by Container Apps Env)
//   * Azure Container Registry (Basic SKU)
//   * Container Apps Managed Environment (Consumption)
//   * Storage Account + Blob container ``models`` for the joblib payload
//   * Two Container Apps with external ingress:
//       - flask    (port 5000, REST API)
//       - streamlit (port 8501, dashboard)
//
// Idle cost:    ~$5/mo (ACR Basic + Log Analytics free tier)
// Active cost:  ~$15-25/mo for a couple of always-on small apps.
// Teardown:     az group delete -n <rg> --yes --no-wait
// =========================================================================

@description('Azure region')
param location string = resourceGroup().location

@description('Short, lowercase project prefix used in resource names')
param projectName string = 'telechurn'

@description('Flask app image (full ACR-qualified tag); set after first ACR build')
param flaskImage string = ''

@description('Streamlit app image (full ACR-qualified tag); set after first ACR build')
param streamlitImage string = ''

// ACR / Storage names need globally-unique strings; uniqueString gives a deterministic 13-char hash.
var nameSuffix    = take(uniqueString(resourceGroup().id), 8)
var acrName       = '${projectName}acr${nameSuffix}'
var storageName   = '${projectName}st${nameSuffix}'
var envName       = '${projectName}-env'
var logName       = '${projectName}-logs'
var flaskAppName  = '${projectName}-flask'
var stAppName     = '${projectName}-streamlit'
var blobContainer = 'models'

// -------------------------------------------------------------------------
// Log Analytics — required by Container Apps Env
// -------------------------------------------------------------------------
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// -------------------------------------------------------------------------
// ACR — hosts the Docker images
// -------------------------------------------------------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: true  // simplest auth path for Container Apps
  }
}

// -------------------------------------------------------------------------
// Storage Account + Blob container ``models``
// -------------------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: true   // joblib is downloaded by the Flask container at startup
    minimumTlsVersion: 'TLS1_2'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: blobContainer
  properties: { publicAccess: 'Blob' }
}

// -------------------------------------------------------------------------
// Container Apps Managed Environment
// -------------------------------------------------------------------------
resource appsEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// -------------------------------------------------------------------------
// Flask Container App
// -------------------------------------------------------------------------
resource flaskApp 'Microsoft.App/containerApps@2024-03-01' = if (!empty(flaskImage)) {
  name: flaskAppName
  location: location
  properties: {
    managedEnvironmentId: appsEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 5000
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.name
          passwordSecretRef: 'acr-pwd'
        }
      ]
      secrets: [
        {
          name: 'acr-pwd'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'flask'
          image: flaskImage
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [
            { name: 'PORT', value: '5000' }
            {
              name: 'AZURE_BLOB_URL'
              value: 'https://${storage.name}.blob.${environment().suffixes.storage}/${blobContainer}/churn_model.joblib'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 5000 }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
      }
    }
  }
}

// -------------------------------------------------------------------------
// Streamlit Container App
// -------------------------------------------------------------------------
resource streamlitApp 'Microsoft.App/containerApps@2024-03-01' = if (!empty(streamlitImage)) {
  name: stAppName
  location: location
  properties: {
    managedEnvironmentId: appsEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.name
          passwordSecretRef: 'acr-pwd'
        }
      ]
      secrets: [
        {
          name: 'acr-pwd'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'streamlit'
          image: streamlitImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'STREAMLIT_SERVER_PORT', value: '8501' }
            { name: 'STREAMLIT_SERVER_ADDRESS', value: '0.0.0.0' }
            { name: 'STREAMLIT_SERVER_HEADLESS', value: 'true' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/_stcore/health', port: 8501 }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
      }
    }
  }
}

// -------------------------------------------------------------------------
// Outputs (consumed by deploy.sh)
// -------------------------------------------------------------------------
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output storageAccountName string = storage.name
output blobContainerName string = blobContainer
output containerAppsEnvName string = appsEnv.name
output flaskAppFqdn string = !empty(flaskImage) ? flaskApp.properties.configuration.ingress.fqdn : ''
output streamlitAppFqdn string = !empty(streamlitImage) ? streamlitApp.properties.configuration.ingress.fqdn : ''
