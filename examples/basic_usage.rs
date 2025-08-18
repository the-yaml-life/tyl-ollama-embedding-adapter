use std::collections::HashMap;
use tyl_embeddings_port::{BatchEmbeddingRequest, ContentType, EmbeddingService};
use tyl_ollama_embedding_adapter::{OllamaConfig, OllamaEmbeddingService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TYL Ollama Embedding Adapter Usage Examples ===\n");

    // Basic usage example
    basic_usage_example().await?;

    // Content-type optimization example
    content_type_optimization_example().await?;

    // Batch processing example
    batch_processing_example().await?;

    // Configuration example
    configuration_example().await?;

    // Health monitoring example
    health_monitoring_example().await?;

    Ok(())
}

async fn basic_usage_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Basic Usage ---");

    // Create Ollama embedding service with default configuration
    let config = OllamaConfig::default();

    println!("Connecting to Ollama at: {}", config.base.service_url);

    // Note: This will fail if Ollama is not running, but shows the usage pattern
    match OllamaEmbeddingService::from_config(config).await {
        Ok(service) => {
            println!("✅ Successfully connected to Ollama service");

            // Try to generate a simple embedding
            let text = "Hello, this is a test embedding";
            match service.generate_embedding(text, ContentType::General).await {
                Ok(embedding) => {
                    println!(
                        "✅ Generated embedding with {} dimensions",
                        embedding.dimensions()
                    );
                    println!("   Model used: {}", embedding.model);
                    println!("   Content type: {:?}", embedding.content_type);
                }
                Err(e) => println!("❌ Failed to generate embedding: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Failed to connect to Ollama: {}", e);
            println!("💡 Make sure Ollama is running on localhost:11434");
        }
    }

    println!();
    Ok(())
}

async fn content_type_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Content-Type Optimization ---");

    let config = OllamaConfig::default();

    // Show how different content types use different models
    println!("Model mapping for different content types:");
    println!(
        "  Code: {}",
        config
            .model_mapping
            .get("code")
            .unwrap_or(&"not configured".to_string())
    );
    println!(
        "  Documentation: {}",
        config
            .model_mapping
            .get("documentation")
            .unwrap_or(&"not configured".to_string())
    );
    println!(
        "  Query: {}",
        config
            .model_mapping
            .get("query")
            .unwrap_or(&"not configured".to_string())
    );

    // Example texts for different content types
    let examples = vec![
        (ContentType::Code, "fn fibonacci(n: u32) -> u32 { if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) } }"),
        (ContentType::Documentation, "This function calculates the Fibonacci sequence using recursion."),
        (ContentType::Query, "how to implement fibonacci in rust"),
        (ContentType::General, "Fibonacci numbers are a sequence in mathematics."),
    ];

    println!("\nExample embeddings for different content types:");

    // Note: This is a demonstration of the API - actual embedding generation would require Ollama
    for (content_type, text) in examples {
        let display_text = if text.len() > 50 {
            format!("{}...", &text[..50])
        } else {
            text.to_string()
        };
        println!("  {:?}: \"{}\"", content_type, display_text);
    }

    println!();
    Ok(())
}

async fn batch_processing_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Batch Processing ---");

    let config = OllamaConfig::default();
    println!("Maximum batch size: {}", config.base.max_batch_size);

    // Create a batch request
    let texts = vec![
        "First document for embedding".to_string(),
        "Second document to process".to_string(),
        "Third piece of content".to_string(),
        "Fourth item in the batch".to_string(),
        "Fifth and final document".to_string(),
    ];

    let batch_request = BatchEmbeddingRequest {
        texts: texts.clone(),
        content_type: ContentType::Documentation,
        model_override: None,
        metadata: HashMap::from([
            ("batch_id".to_string(), "example_batch_001".to_string()),
            ("source".to_string(), "documentation".to_string()),
        ]),
    };

    println!("Batch request created:");
    println!("  Texts count: {}", batch_request.texts.len());
    println!("  Content type: {:?}", batch_request.content_type);
    println!("  Metadata: {:?}", batch_request.metadata);

    // Show chunking for large batches
    if texts.len() > config.base.max_batch_size {
        println!(
            "⚠️  This batch would be chunked into {} smaller batches",
            (texts.len() + config.base.max_batch_size - 1) / config.base.max_batch_size
        );
    } else {
        println!("✅ This batch fits within the maximum batch size");
    }

    println!();
    Ok(())
}

async fn configuration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Configuration Options ---");

    let config = OllamaConfig::default();

    println!("Ollama Configuration:");
    println!("  Host: {}", config.host);
    println!("  Port: {}", config.port);
    println!("  Service URL: {}", config.base.service_url);
    println!("  Default model: {}", config.base.model);
    println!("  Connection pool size: {}", config.connection_pool_size);
    println!("  Auto-pull models: {}", config.auto_pull_models);
    println!("  Retry attempts: {}", config.retry_attempts);
    println!("  Retry delay: {}ms", config.retry_delay_ms);
    println!("  Request queue size: {}", config.request_queue_size);

    println!("\nAvailable models:");
    for model in &config.available_models {
        println!("  - {}", model);
    }

    println!("\nEnvironment variables you can set:");
    println!("  TYL_OLLAMA_HOST - Ollama server host");
    println!("  TYL_OLLAMA_PORT - Ollama server port");
    println!("  TYL_OLLAMA_CONNECTION_POOL_SIZE - HTTP connection pool size");
    println!("  TYL_OLLAMA_AUTO_PULL_MODELS - Auto-pull missing models (true/false)");
    println!("  TYL_OLLAMA_RETRY_ATTEMPTS - Number of retry attempts");
    println!("  TYL_OLLAMA_CODE_MODEL - Model for code content");
    println!("  TYL_OLLAMA_TEXT_MODEL - Model for text content");
    println!("  TYL_OLLAMA_AVAILABLE_MODELS - Comma-separated list of available models");

    println!();
    Ok(())
}

async fn health_monitoring_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Health Monitoring ---");

    let config = OllamaConfig::default();

    println!("Health check configuration:");
    println!("  Timeout: {} seconds", config.base.timeout_seconds);
    println!("  Service URL: {}", config.base.service_url);

    // Example of what a health check would return
    println!("\nHealth check would verify:");
    println!("  ✓ Ollama service is accessible");
    println!("  ✓ Required models are available");
    println!("  ✓ Service can generate test embeddings");

    // Note: Actual health check would require Ollama to be running
    println!("\n💡 To test health check, ensure Ollama is running and try:");
    println!("   cargo run --example basic_usage");

    println!();
    Ok(())
}
