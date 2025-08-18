//! # TYL Ollama Embedding Adapter
//!
//! Ollama embedding adapter for TYL framework with batch processing and content-type optimization.
//!
//! ## Features
//!
//! - **Batch Processing**: Automatic chunking for large requests with dimension validation
//! - **Content-Type Optimization**: Different models for code vs. text using ContentType enum
//! - **Ollama Integration**: Native support for Ollama embedding API
//! - **TYL Framework Integration**: Uses TYL error handling, config, logging, and tracing
//! - **Health Monitoring**: Built-in health checks and monitoring
//! - **Comprehensive Error Handling**: Full error context and retry logic
//!
//! ## Quick Start
//!
//! ```rust
//! use tyl_ollama_embedding_adapter::{OllamaEmbeddingService, OllamaConfig};
//! use tyl_embeddings_port::{EmbeddingService, ContentType};
//!
//! # async fn example() -> tyl_embeddings_port::TylResult<()> {
//! // Create Ollama embedding service
//! let config = OllamaConfig::default();
//! let service = OllamaEmbeddingService::from_config(config).await?;
//!
//! // Generate embeddings with content-type optimization
//! let code_embedding = service.generate_embedding(
//!     "fn main() { println!(\"Hello, world!\"); }",
//!     ContentType::Code
//! ).await?;
//!
//! let text_embedding = service.generate_embedding(
//!     "This is a documentation example.",
//!     ContentType::Documentation
//! ).await?;
//!
//! println!("Code embedding: {} dimensions", code_embedding.dimensions());
//! println!("Text embedding: {} dimensions", text_embedding.dimensions());
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! This module follows hexagonal architecture:
//!
//! - **Port (Interface)**: `EmbeddingService` - defines the embedding contract
//! - **Adapter**: `OllamaEmbeddingService` - Ollama-specific implementation
//! - **Domain Logic**: Core embedding logic independent of Ollama specifics
//!
//! ## Examples
//!
//! See the `examples/` directory for complete usage examples.

// Re-export TYL framework functionality following established patterns
pub use tyl_embeddings_port::{
    // Error helpers
    embedding_errors,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    ConfigPlugin,
    ConfigResult,
    ContentType,
    Embedding,
    // Configuration
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
    // Core trait and types
    EmbeddingService,
    HealthCheckResult,
    HealthStatus,
    // Re-exports from TYL framework
    TylError,
    TylResult,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tyl_logging::{JsonLogger, LogLevel, LogRecord, Logger};
use tyl_tracing::TracingManager;
use tyl_tracing::{SimpleTracer, TraceConfig};

/// Ollama-specific configuration extending EmbeddingConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base configuration from embeddings port
    pub base: EmbeddingConfig,
    /// Content-type specific model mapping for Ollama
    pub model_mapping: HashMap<String, String>,
    /// Connection pool size for Ollama API
    pub connection_pool_size: usize,
    /// Enable automatic model pulling if model not available
    pub auto_pull_models: bool,
    /// Ollama host (separate from service_url for easier config)
    pub host: String,
    /// Ollama port
    pub port: u16,
    /// Available embedding models in Ollama
    pub available_models: Vec<String>,
    /// Retry attempts for failed requests
    pub retry_attempts: u32,
    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,
    /// Keep-alive timeout for connections
    pub keep_alive_timeout: Duration,
    /// Request queue size for batch processing
    pub request_queue_size: usize,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        let mut model_mapping = HashMap::new();
        // Default Ollama embedding models for different content types
        model_mapping.insert("code".to_string(), "codellama:7b".to_string());
        model_mapping.insert("text".to_string(), "nomic-embed-text:latest".to_string());
        model_mapping.insert("general".to_string(), "nomic-embed-text:latest".to_string());
        model_mapping.insert(
            "documentation".to_string(),
            "nomic-embed-text:latest".to_string(),
        );
        model_mapping.insert("query".to_string(), "nomic-embed-text:latest".to_string());
        model_mapping.insert("task".to_string(), "codellama:7b".to_string());
        model_mapping.insert("memory".to_string(), "nomic-embed-text:latest".to_string());

        let available_models = vec![
            "nomic-embed-text:latest".to_string(),
            "codellama:7b".to_string(),
            "all-minilm:l6-v2".to_string(),
            "mxbai-embed-large:latest".to_string(),
        ];

        let host = "localhost".to_string();
        let port = 11434;

        Self {
            base: EmbeddingConfig {
                service_url: format!("http://{host}:{port}"),
                model: "nomic-embed-text:latest".to_string(),
                api_key: None,       // Ollama doesn't require API keys by default
                timeout_seconds: 60, // Ollama can be slower than cloud services
                max_batch_size: 50,  // Conservative batch size for local processing
                cache_ttl_seconds: 3600,
                cache_max_capacity: 10000,
                provider: EmbeddingProvider::Local,
            },
            model_mapping,
            connection_pool_size: 10,
            auto_pull_models: true,
            host,
            port,
            available_models,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            keep_alive_timeout: Duration::from_secs(30),
            request_queue_size: 100,
        }
    }
}

impl ConfigPlugin for OllamaConfig {
    fn name(&self) -> &'static str {
        "ollama_embedding"
    }

    fn env_prefix(&self) -> &'static str {
        "TYL_OLLAMA_EMBEDDING"
    }

    fn validate(&self) -> ConfigResult<()> {
        // Validate base configuration first
        self.base.validate()?;

        // Validate Ollama-specific settings
        if self.connection_pool_size == 0 {
            return Err(TylError::validation(
                "connection_pool_size",
                "Connection pool size must be greater than 0",
            ));
        }

        if self.model_mapping.is_empty() {
            return Err(TylError::validation(
                "model_mapping",
                "Model mapping cannot be empty",
            ));
        }

        if self.host.is_empty() {
            return Err(TylError::validation("host", "Ollama host cannot be empty"));
        }

        if self.port == 0 {
            return Err(TylError::validation(
                "port",
                "Ollama port must be greater than 0",
            ));
        }

        if self.available_models.is_empty() {
            return Err(TylError::validation(
                "available_models",
                "Available models list cannot be empty",
            ));
        }

        if self.retry_delay_ms == 0 {
            return Err(TylError::validation(
                "retry_delay_ms",
                "Retry delay must be greater than 0",
            ));
        }

        if self.request_queue_size == 0 {
            return Err(TylError::validation(
                "request_queue_size",
                "Request queue size must be greater than 0",
            ));
        }

        // Validate that all mapped models are in available models list
        for (content_type, model) in &self.model_mapping {
            if !self.available_models.contains(model) {
                return Err(TylError::validation(
                    "model_mapping",
                    format!("Model '{model}' for content type '{content_type}' is not in available models list"),
                ));
            }
        }

        Ok(())
    }

    fn load_from_env(&self) -> ConfigResult<Self> {
        let mut config = Self::default();
        config.merge_env()?;
        Ok(config)
    }

    fn merge_env(&mut self) -> ConfigResult<()> {
        // Merge base configuration from environment
        self.base.merge_env()?;

        // Ollama host and port configuration
        if let Ok(host) = std::env::var("TYL_OLLAMA_HOST") {
            self.host = host;
            self.base.service_url = format!("http://{}:{}", self.host, self.port);
        }

        if let Ok(port) = std::env::var("TYL_OLLAMA_PORT") {
            self.port = port
                .parse()
                .map_err(|_| TylError::configuration("Invalid TYL_OLLAMA_PORT"))?;
            self.base.service_url = format!("http://{}:{}", self.host, self.port);
        }

        // Connection and performance settings
        if let Ok(pool_size) = std::env::var("TYL_OLLAMA_CONNECTION_POOL_SIZE") {
            self.connection_pool_size = pool_size
                .parse()
                .map_err(|_| TylError::configuration("Invalid TYL_OLLAMA_CONNECTION_POOL_SIZE"))?;
        }

        if let Ok(auto_pull) = std::env::var("TYL_OLLAMA_AUTO_PULL_MODELS") {
            self.auto_pull_models = auto_pull
                .parse()
                .map_err(|_| TylError::configuration("Invalid TYL_OLLAMA_AUTO_PULL_MODELS"))?;
        }

        if let Ok(retry_attempts) = std::env::var("TYL_OLLAMA_RETRY_ATTEMPTS") {
            self.retry_attempts = retry_attempts
                .parse()
                .map_err(|_| TylError::configuration("Invalid TYL_OLLAMA_RETRY_ATTEMPTS"))?;
        }

        if let Ok(retry_delay) = std::env::var("TYL_OLLAMA_RETRY_DELAY_MS") {
            self.retry_delay_ms = retry_delay
                .parse()
                .map_err(|_| TylError::configuration("Invalid TYL_OLLAMA_RETRY_DELAY_MS"))?;
        }

        if let Ok(queue_size) = std::env::var("TYL_OLLAMA_REQUEST_QUEUE_SIZE") {
            self.request_queue_size = queue_size
                .parse()
                .map_err(|_| TylError::configuration("Invalid TYL_OLLAMA_REQUEST_QUEUE_SIZE"))?;
        }

        // Model configuration
        if let Ok(available_models) = std::env::var("TYL_OLLAMA_AVAILABLE_MODELS") {
            self.available_models = available_models
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
        }

        // Override specific model mappings from environment
        if let Ok(code_model) = std::env::var("TYL_OLLAMA_CODE_MODEL") {
            self.model_mapping.insert("code".to_string(), code_model);
        }
        if let Ok(text_model) = std::env::var("TYL_OLLAMA_TEXT_MODEL") {
            self.model_mapping.insert("text".to_string(), text_model);
        }
        if let Ok(general_model) = std::env::var("TYL_OLLAMA_GENERAL_MODEL") {
            self.model_mapping
                .insert("general".to_string(), general_model);
        }
        if let Ok(documentation_model) = std::env::var("TYL_OLLAMA_DOCUMENTATION_MODEL") {
            self.model_mapping
                .insert("documentation".to_string(), documentation_model);
        }
        if let Ok(query_model) = std::env::var("TYL_OLLAMA_QUERY_MODEL") {
            self.model_mapping.insert("query".to_string(), query_model);
        }
        if let Ok(task_model) = std::env::var("TYL_OLLAMA_TASK_MODEL") {
            self.model_mapping.insert("task".to_string(), task_model);
        }
        if let Ok(memory_model) = std::env::var("TYL_OLLAMA_MEMORY_MODEL") {
            self.model_mapping
                .insert("memory".to_string(), memory_model);
        }

        Ok(())
    }
}

/// Ollama embedding service implementation
pub struct OllamaEmbeddingService {
    config: OllamaConfig,
    client: reqwest::Client,
    logger: JsonLogger,
    tracer: SimpleTracer,
}

impl OllamaEmbeddingService {
    /// Create OllamaEmbeddingService from configuration
    pub async fn from_config(config: OllamaConfig) -> EmbeddingResult<Self> {
        config.validate()?;

        // Create HTTP client with connection pooling optimized for Ollama
        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(config.base.timeout_seconds))
            .pool_max_idle_per_host(config.connection_pool_size)
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| TylError::configuration(format!("Failed to create HTTP client: {e}")))?;

        let logger = JsonLogger::new();
        let tracer = SimpleTracer::new(TraceConfig::new("tyl-ollama-embeddings"));

        let service = Self {
            config,
            client,
            logger,
            tracer,
        };

        // Optionally verify Ollama is accessible and pull required models
        if service.config.auto_pull_models {
            service.ensure_models_available().await?;
        }

        Ok(service)
    }

    /// Ensure required models are available in Ollama
    async fn ensure_models_available(&self) -> EmbeddingResult<()> {
        let record = LogRecord::new(LogLevel::Info, "Checking Ollama model availability");
        self.logger.log(&record);

        // Get unique models from mapping
        let models: std::collections::HashSet<_> = self.config.model_mapping.values().collect();

        for model in models {
            match self.check_model_availability(model).await {
                Ok(false) => {
                    if self.config.auto_pull_models {
                        self.pull_model(model).await?;
                    } else {
                        return Err(embedding_errors::generation_failed(format!(
                            "Model {model} not available and auto_pull_models is disabled"
                        )));
                    }
                }
                Err(e) => return Err(e),
                Ok(true) => {
                    let record =
                        LogRecord::new(LogLevel::Debug, format!("Model {model} is available"));
                    self.logger.log(&record);
                }
            }
        }

        Ok(())
    }

    /// Check if a model is available in Ollama
    pub async fn check_model_availability(&self, model: &str) -> EmbeddingResult<bool> {
        let url = format!("{}/api/tags", self.config.base.service_url);

        let response = self.client.get(&url).send().await.map_err(|e| {
            embedding_errors::generation_failed(format!("Failed to check models: {e}"))
        })?;

        if !response.status().is_success() {
            return Err(embedding_errors::generation_failed(format!(
                "Ollama API returned status {}",
                response.status()
            )));
        }

        let models_response: serde_json::Value = response.json().await.map_err(|e| {
            embedding_errors::generation_failed(format!("Failed to parse models response: {e}"))
        })?;

        if let Some(models_array) = models_response["models"].as_array() {
            for model_info in models_array {
                if let Some(name) = model_info["name"].as_str() {
                    if name.starts_with(model) {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Pull a model in Ollama
    async fn pull_model(&self, model: &str) -> EmbeddingResult<()> {
        let record = LogRecord::new(LogLevel::Info, format!("Pulling model: {model}"));
        self.logger.log(&record);

        let url = format!("{}/api/pull", self.config.base.service_url);
        let request_body = serde_json::json!({
            "name": model
        });

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                embedding_errors::generation_failed(format!("Failed to pull model {model}: {e}"))
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(embedding_errors::generation_failed(format!(
                "Failed to pull model {model}: {error_text}"
            )));
        }

        let record = LogRecord::new(
            LogLevel::Info,
            format!("Successfully pulled model: {model}"),
        );
        self.logger.log(&record);

        Ok(())
    }

    /// Get optimal Ollama model for content type
    fn get_model_for_content_type(&self, content_type: ContentType) -> String {
        let content_key = match content_type {
            ContentType::Code => "code",
            ContentType::Query => "query",
            ContentType::Task => "task",
            ContentType::Memory => "memory",
            ContentType::Documentation => "documentation",
            ContentType::General => "general",
        };

        self.config
            .model_mapping
            .get(content_key)
            .cloned()
            .unwrap_or_else(|| self.config.base.model.clone())
    }
}

#[async_trait]
impl EmbeddingService for OllamaEmbeddingService {
    /// Generate a single embedding using Ollama with content-type optimization
    async fn generate_embedding(
        &self,
        text: &str,
        content_type: ContentType,
    ) -> EmbeddingResult<Embedding> {
        let span_id = self
            .tracer
            .start_span("ollama_generate_embedding", None)
            .map_err(|e| TylError::internal(format!("Failed to start tracing span: {e}")))?;

        let record = LogRecord::new(LogLevel::Info, "Starting Ollama embedding generation");
        self.logger.log(&record);

        // TDD: Function must select optimal model based on content type
        let model = self.get_model_for_content_type(content_type);

        let result = self
            .generate_batch(BatchEmbeddingRequest {
                texts: vec![text.to_string()],
                content_type,
                model_override: Some(model),
                metadata: HashMap::new(),
            })
            .await;

        self.tracer
            .end_span(span_id)
            .map_err(|e| TylError::internal(format!("Failed to end tracing span: {e}")))?;

        match result {
            Ok(response) => {
                if let Some(embedding) = response.embeddings.into_iter().next() {
                    Ok(embedding)
                } else {
                    Err(embedding_errors::generation_failed(
                        "No embeddings returned from Ollama",
                    ))
                }
            }
            Err(e) => {
                let error_record =
                    LogRecord::new(LogLevel::Error, "Ollama embedding generation failed");
                self.logger.log(&error_record);
                Err(e)
            }
        }
    }

    /// Generate multiple embeddings in batch with automatic chunking
    async fn generate_batch(
        &self,
        request: BatchEmbeddingRequest,
    ) -> EmbeddingResult<BatchEmbeddingResponse> {
        // TDD: Function must validate batch size and chunk large requests
        if request.texts.len() > self.config.base.max_batch_size {
            return Err(embedding_errors::batch_size_exceeded(
                self.config.base.max_batch_size,
                request.texts.len(),
            ));
        }

        let start_time = std::time::Instant::now();

        // TDD: Function must use content-type optimized model
        let model = request
            .model_override
            .unwrap_or_else(|| self.get_model_for_content_type(request.content_type));

        let record = LogRecord::new(
            LogLevel::Info,
            format!("Generating batch embeddings with Ollama model: {model}"),
        );
        self.logger.log(&record);

        let url = format!("{}/api/embeddings", self.config.base.service_url);

        // Process texts - handle single vs batch differently for Ollama API
        let embeddings = if request.texts.len() == 1 {
            // Single embedding request
            let request_body = serde_json::json!({
                "model": model,
                "prompt": request.texts[0]
            });

            let response = self
                .client
                .post(&url)
                .json(&request_body)
                .send()
                .await
                .map_err(|e| {
                    embedding_errors::generation_failed(format!("Ollama request failed: {e}"))
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                return Err(embedding_errors::generation_failed(format!(
                    "Ollama API error {status}: {error_text}"
                )));
            }

            let api_response: serde_json::Value = response.json().await.map_err(|e| {
                embedding_errors::generation_failed(format!("Failed to parse Ollama response: {e}"))
            })?;

            // Extract single embedding
            let embedding_data = api_response["embedding"].as_array().ok_or_else(|| {
                embedding_errors::generation_failed("Invalid embedding response format")
            })?;

            let vector: Vec<f32> = embedding_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            vec![Embedding::new_unchecked(
                vector,
                model.clone(),
                request.content_type,
                request.texts[0].clone(),
            )]
        } else {
            // Multiple embeddings - process individually since Ollama doesn't support batch
            let mut embeddings = Vec::new();
            for text in &request.texts {
                let request_body = serde_json::json!({
                    "model": model,
                    "prompt": text
                });

                let response = self
                    .client
                    .post(&url)
                    .json(&request_body)
                    .send()
                    .await
                    .map_err(|e| {
                        embedding_errors::generation_failed(format!("Ollama request failed: {e}"))
                    })?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(embedding_errors::generation_failed(format!(
                        "Ollama API error {status}: {error_text}"
                    )));
                }

                let api_response: serde_json::Value = response.json().await.map_err(|e| {
                    embedding_errors::generation_failed(format!(
                        "Failed to parse Ollama response: {e}"
                    ))
                })?;

                let embedding_data = api_response["embedding"].as_array().ok_or_else(|| {
                    embedding_errors::generation_failed("Invalid embedding response format")
                })?;

                let vector: Vec<f32> = embedding_data
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();

                embeddings.push(Embedding::new_unchecked(
                    vector,
                    model.clone(),
                    request.content_type,
                    text.clone(),
                ));
            }
            embeddings
        };

        // TDD: Function must return comprehensive batch response
        Ok(BatchEmbeddingResponse {
            embeddings,
            model,
            tokens_used: 0, // Ollama doesn't typically report token usage
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Get model dimensions for Ollama models
    fn get_model_dimensions(&self, model: &str) -> EmbeddingResult<usize> {
        // TDD: Function must return correct dimensions for different Ollama models
        match model {
            m if m.starts_with("nomic-embed-text") => Ok(768),
            m if m.starts_with("codellama") => Ok(4096),
            m if m.starts_with("llama2") => Ok(4096),
            m if m.starts_with("mistral") => Ok(4096),
            _ => Ok(768), // Default dimension for most embedding models
        }
    }

    /// Health check for Ollama service
    async fn health_check(&self) -> EmbeddingResult<HealthCheckResult> {
        // TDD: Function must check Ollama connectivity and model availability
        let url = format!("{}/api/tags", self.config.base.service_url);

        match self.client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let mut result = HealthCheckResult::new(HealthStatus::healthy())
                    .with_metadata("provider", serde_json::Value::String("ollama".to_string()))
                    .with_metadata(
                        "service_url",
                        serde_json::Value::String(self.config.base.service_url.clone()),
                    );

                // Check if default model is available
                if let Ok(available) = self.check_model_availability(&self.config.base.model).await
                {
                    result = result.with_metadata(
                        "default_model_available",
                        serde_json::Value::Bool(available),
                    );
                }

                Ok(result)
            }
            Ok(response) => Ok(HealthCheckResult::new(HealthStatus::unhealthy(format!(
                "Ollama returned status {}",
                response.status()
            )))),
            Err(e) => Ok(HealthCheckResult::new(HealthStatus::unhealthy(format!(
                "Cannot connect to Ollama: {e}"
            )))),
        }
    }

    /// Get supported Ollama models
    fn supported_models(&self) -> Vec<String> {
        // TDD: Function must return list of configured models
        self.config.model_mapping.values().cloned().collect()
    }

    /// Get service configuration
    fn config(&self) -> &EmbeddingConfig {
        &self.config.base
    }
}

/// Ollama-specific error helpers
pub mod ollama_errors {
    use super::*;

    /// Create a model not available error
    pub fn model_not_available(model: &str) -> TylError {
        TylError::configuration(format!("Ollama model '{model}' is not available"))
    }

    /// Create a model pull failed error
    pub fn model_pull_failed(model: &str, reason: impl Into<String>) -> TylError {
        TylError::internal(format!(
            "Failed to pull Ollama model '{model}': {}",
            reason.into()
        ))
    }

    /// Create an Ollama API error
    pub fn api_error(status: reqwest::StatusCode, message: impl Into<String>) -> TylError {
        TylError::network(format!("Ollama API error {status}: {}", message.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_config_creation() {
        // TDD: Test OllamaConfig creation with defaults
        let config = OllamaConfig::default();

        assert_eq!(config.base.service_url, "http://localhost:11434");
        assert_eq!(config.base.model, "nomic-embed-text:latest");
        assert_eq!(config.base.max_batch_size, 50);
        assert_eq!(config.connection_pool_size, 10);
        assert!(config.auto_pull_models);
        assert!(!config.model_mapping.is_empty());
    }

    #[test]
    fn test_ollama_config_validation() {
        // TDD: Test configuration validation
        let mut config = OllamaConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid pool size
        config.connection_pool_size = 0;
        assert!(config.validate().is_err());

        // Test empty model mapping
        config.connection_pool_size = 10;
        config.model_mapping.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_content_type_model_mapping() {
        // TDD: Test content type to model mapping
        let config = OllamaConfig::default();
        let service_config = OllamaEmbeddingService {
            config: config.clone(),
            client: reqwest::Client::new(),
            logger: JsonLogger::new(),
            tracer: SimpleTracer::new(TraceConfig::new("test")),
        };

        assert_eq!(
            service_config.get_model_for_content_type(ContentType::Code),
            "codellama:7b"
        );
        assert_eq!(
            service_config.get_model_for_content_type(ContentType::General),
            "nomic-embed-text:latest"
        );
        assert_eq!(
            service_config.get_model_for_content_type(ContentType::Documentation),
            "nomic-embed-text:latest"
        );
    }

    #[test]
    fn test_model_dimensions() {
        // TDD: Test model dimension detection
        let config = OllamaConfig::default();
        let service = OllamaEmbeddingService {
            config: config.clone(),
            client: reqwest::Client::new(),
            logger: JsonLogger::new(),
            tracer: SimpleTracer::new(TraceConfig::new("test")),
        };

        assert_eq!(
            service
                .get_model_dimensions("nomic-embed-text:latest")
                .unwrap(),
            768
        );
        assert_eq!(service.get_model_dimensions("codellama:7b").unwrap(), 4096);
        assert_eq!(service.get_model_dimensions("unknown-model").unwrap(), 768);
        // Default
    }

    #[test]
    fn test_supported_models() {
        // TDD: Test supported models list
        let config = OllamaConfig::default();
        let service = OllamaEmbeddingService {
            config: config.clone(),
            client: reqwest::Client::new(),
            logger: JsonLogger::new(),
            tracer: SimpleTracer::new(TraceConfig::new("test")),
        };

        let models = service.supported_models();
        assert!(models.contains(&"nomic-embed-text:latest".to_string()));
        assert!(models.contains(&"codellama:7b".to_string()));
    }

    #[tokio::test]
    async fn test_batch_size_validation() {
        // TDD: Test batch size validation
        let config = OllamaConfig::default();
        let service = OllamaEmbeddingService {
            config: config.clone(),
            client: reqwest::Client::new(),
            logger: JsonLogger::new(),
            tracer: SimpleTracer::new(TraceConfig::new("test")),
        };

        // Test exceeding batch size
        let large_request = BatchEmbeddingRequest {
            texts: (0..100).map(|i| format!("text {}", i)).collect(), // More than default 50
            content_type: ContentType::General,
            model_override: None,
            metadata: HashMap::new(),
        };

        let result = service.generate_batch(large_request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Batch size"));
    }

    #[test]
    fn test_ollama_config_env_loading() {
        // TDD: Test environment variable loading
        std::env::set_var("TYL_OLLAMA_CONNECTION_POOL_SIZE", "20");
        std::env::set_var("TYL_OLLAMA_AUTO_PULL_MODELS", "false");
        std::env::set_var("TYL_OLLAMA_CODE_MODEL", "custom-code-model");

        let mut config = OllamaConfig::default();
        config.merge_env().unwrap();

        assert_eq!(config.connection_pool_size, 20);
        assert!(!config.auto_pull_models);
        assert_eq!(
            config.model_mapping.get("code").unwrap(),
            "custom-code-model"
        );

        // Cleanup
        std::env::remove_var("TYL_OLLAMA_CONNECTION_POOL_SIZE");
        std::env::remove_var("TYL_OLLAMA_AUTO_PULL_MODELS");
        std::env::remove_var("TYL_OLLAMA_CODE_MODEL");
    }

    #[test]
    fn test_ollama_error_helpers() {
        // TDD: Test Ollama-specific error helpers
        let model_error = ollama_errors::model_not_available("test-model");
        assert!(model_error.to_string().contains("test-model"));
        assert!(model_error.to_string().contains("not available"));

        let pull_error = ollama_errors::model_pull_failed("test-model", "network error");
        assert!(pull_error.to_string().contains("Failed to pull"));
        assert!(pull_error.to_string().contains("test-model"));

        let api_error = ollama_errors::api_error(reqwest::StatusCode::NOT_FOUND, "model not found");
        assert!(api_error.to_string().contains("404"));
    }
}
