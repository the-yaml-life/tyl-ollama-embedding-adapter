use std::collections::HashMap;
use tyl_embeddings_port::{BatchEmbeddingRequest, ContentType, EmbeddingService};
use tyl_ollama_embedding_adapter::{OllamaConfig, OllamaEmbeddingService};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Integration tests for Ollama embedding adapter
/// These tests work in two modes:
/// 1. If Ollama is available on ishtar:11434 -> Use real Ollama
/// 2. If Ollama is not available (CI) -> Use mock server

/// Check if real Ollama is available on ishtar
async fn is_ollama_available() -> bool {
    let client = reqwest::Client::new();
    match client.get("http://ishtar:11434/api/tags").send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Set up mock Ollama server for CI
async fn setup_mock_ollama() -> MockServer {
    let mock_server = MockServer::start().await;

    // Mock /api/tags endpoint (model list)
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                {
                    "name": "nomic-embed-text:latest",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 274301056,
                    "digest": "mock-digest"
                }
            ]
        })))
        .mount(&mock_server)
        .await;

    // Mock /api/embeddings endpoint
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embedding": vec![0.1; 768] // Mock 768-dimensional embedding
        })))
        .mount(&mock_server)
        .await;

    mock_server
}

/// Create appropriate config (real Ollama or mock)
async fn create_test_config() -> (OllamaConfig, Option<MockServer>) {
    if is_ollama_available().await {
        println!("🚀 Using real Ollama on ishtar:11434");
        let mut config = OllamaConfig::default();
        config.host = "ishtar".to_string();
        config.port = 11434;
        config.base.service_url = format!("http://{}:{}", config.host, config.port);
        config.auto_pull_models = false;
        (config, None)
    } else {
        println!("🤖 Using mock Ollama server for CI");
        let mock_server = setup_mock_ollama().await;
        let mut config = OllamaConfig::default();
        config.base.service_url = mock_server.uri();
        config.auto_pull_models = false;
        (config, Some(mock_server))
    }
}

#[tokio::test]
async fn test_ollama_connectivity_ishtar() {
    // TDD: Test basic connectivity to Ollama (real or mock)
    let (config, _mock_server) = create_test_config().await;

    println!(
        "Testing connectivity to Ollama at: {}",
        config.base.service_url
    );

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama (real or mock)");

    println!("✅ Successfully connected to Ollama");

    // Test health check
    let health = service
        .health_check()
        .await
        .expect("Health check should succeed");

    println!("Health check status: {:?}", health.status);
    assert!(health.status.is_healthy(), "Ollama should be healthy");
}

#[tokio::test]
async fn test_nomic_embed_text_model_available() {
    // TDD: Test that nomic-embed-text:latest model is available
    let (mut config, _mock_server) = create_test_config().await;
    config.base.model = "nomic-embed-text:latest".to_string();

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama (real or mock)");

    // Check if the specific model is available
    let available = service
        .check_model_availability("nomic-embed-text:latest")
        .await
        .expect("Should be able to check model availability");

    assert!(
        available,
        "nomic-embed-text:latest model should be available on ishtar"
    );
    println!("✅ nomic-embed-text:latest model is available on ishtar");
}

#[tokio::test]
async fn test_single_embedding_generation_ishtar() {
    // TDD: Test single embedding generation (real Ollama or mock)
    let (mut config, _mock_server) = create_test_config().await;
    config.base.model = "nomic-embed-text:latest".to_string();

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama (real or mock)");

    let test_text = "This is a test sentence for embedding generation.";
    println!("Generating embedding for: \"{}\"", test_text);

    let embedding = service
        .generate_embedding(test_text, ContentType::General)
        .await
        .expect("Should generate embedding successfully");

    println!("✅ Generated embedding:");
    println!("   Dimensions: {}", embedding.dimensions());
    println!("   Model: {}", embedding.model);
    println!("   Content type: {:?}", embedding.content_type);
    println!("   Source text: {}", embedding.source_text);

    // Validate embedding properties
    assert!(
        embedding.dimensions() > 0,
        "Embedding should have dimensions"
    );
    assert_eq!(embedding.model, "nomic-embed-text:latest");
    assert_eq!(embedding.content_type, ContentType::General);
    assert_eq!(embedding.source_text, test_text);
    assert!(!embedding.vector.is_empty(), "Vector should not be empty");
}

#[tokio::test]
async fn test_content_type_optimization_ishtar() {
    // TDD: Test content-type optimization with different text types
    let (mut config, _mock_server) = create_test_config().await;

    // Override model mapping to use only available model
    config
        .model_mapping
        .insert("code".to_string(), "nomic-embed-text:latest".to_string());
    config.model_mapping.insert(
        "documentation".to_string(),
        "nomic-embed-text:latest".to_string(),
    );
    config
        .model_mapping
        .insert("general".to_string(), "nomic-embed-text:latest".to_string());

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama on ishtar");

    let test_cases = vec![
        (
            ContentType::Code,
            "fn main() { println!(\"Hello, world!\"); }",
        ),
        (
            ContentType::Documentation,
            "This function prints a greeting message to the console.",
        ),
        (
            ContentType::General,
            "Hello world is a traditional first program.",
        ),
    ];

    for (content_type, text) in test_cases {
        println!("Testing {:?} content with text: \"{}\"", content_type, text);

        let embedding = service
            .generate_embedding(text, content_type)
            .await
            .expect(&format!(
                "Should generate embedding for {:?} content",
                content_type
            ));

        println!("✅ {:?} embedding generated:", content_type);
        println!("   Dimensions: {}", embedding.dimensions());
        println!("   Model: {}", embedding.model);

        assert!(embedding.dimensions() > 0);
        assert_eq!(embedding.content_type, content_type);
        assert_eq!(embedding.source_text, text);
        assert_eq!(embedding.model, "nomic-embed-text:latest");
    }
}

#[tokio::test]
async fn test_batch_processing_ishtar() {
    // TDD: Test batch processing (real Ollama or mock)
    let (mut config, _mock_server) = create_test_config().await;
    config.base.model = "nomic-embed-text:latest".to_string();
    config.base.max_batch_size = 10; // Small batch for testing

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama on ishtar");

    let test_texts = vec![
        "First document for batch processing".to_string(),
        "Second document in the batch".to_string(),
        "Third piece of content to embed".to_string(),
        "Fourth item for testing batches".to_string(),
        "Fifth and final document".to_string(),
    ];

    let batch_request = BatchEmbeddingRequest {
        texts: test_texts.clone(),
        content_type: ContentType::Documentation,
        model_override: Some("nomic-embed-text:latest".to_string()),
        metadata: HashMap::from([
            ("test_id".to_string(), "integration_batch_001".to_string()),
            ("environment".to_string(), "ishtar".to_string()),
        ]),
    };

    println!("Processing batch of {} texts", batch_request.texts.len());

    let start_time = std::time::Instant::now();
    let batch_response = service
        .generate_batch(batch_request)
        .await
        .expect("Should process batch successfully");
    let processing_time = start_time.elapsed();

    println!("✅ Batch processing completed:");
    println!(
        "   Embeddings generated: {}",
        batch_response.embeddings.len()
    );
    println!("   Model used: {}", batch_response.model);
    println!("   Processing time: {:?}", processing_time);
    println!(
        "   Reported processing time: {}ms",
        batch_response.processing_time_ms
    );

    // Validate batch response
    assert_eq!(batch_response.embeddings.len(), test_texts.len());
    assert_eq!(batch_response.model, "nomic-embed-text:latest");

    // Validate each embedding in the batch
    for (i, embedding) in batch_response.embeddings.iter().enumerate() {
        assert!(
            embedding.dimensions() > 0,
            "Embedding {} should have dimensions",
            i
        );
        assert_eq!(embedding.source_text, test_texts[i]);
        assert_eq!(embedding.content_type, ContentType::Documentation);
        assert_eq!(embedding.model, "nomic-embed-text:latest");
    }
}

#[tokio::test]
async fn test_model_dimensions_detection_ishtar() {
    // TDD: Test model dimension detection for nomic-embed-text (real Ollama or mock)
    let (config, _mock_server) = create_test_config().await;

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama on ishtar");

    let dimensions = service
        .get_model_dimensions("nomic-embed-text:latest")
        .expect("Should get dimensions for nomic-embed-text");

    println!("✅ nomic-embed-text:latest dimensions: {}", dimensions);
    assert!(dimensions > 0, "Dimensions should be greater than 0");

    // nomic-embed-text typically has 768 dimensions
    assert_eq!(
        dimensions, 768,
        "nomic-embed-text should have 768 dimensions"
    );
}

#[tokio::test]
async fn test_supported_models_ishtar() {
    // TDD: Test that supported models list is accurate (real Ollama or mock)
    let (config, _mock_server) = create_test_config().await;

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama on ishtar");

    let supported_models = service.supported_models();
    println!("✅ Supported models: {:?}", supported_models);

    assert!(!supported_models.is_empty(), "Should have supported models");
    assert!(
        supported_models.contains(&"nomic-embed-text:latest".to_string()),
        "Should include nomic-embed-text:latest in supported models"
    );
}

#[tokio::test]
async fn test_error_handling_ishtar() {
    // TDD: Test error handling with invalid configurations (real Ollama or mock)
    let (config, _mock_server) = create_test_config().await;

    let service = OllamaEmbeddingService::from_config(config)
        .await
        .expect("Should connect to Ollama on ishtar");

    // Test with empty text
    let result = service.generate_embedding("", ContentType::General).await;
    println!("Empty text result: {:?}", result);
    // Note: Ollama might handle empty strings differently, so we just log the result

    // Test batch size exceeded
    let large_batch = BatchEmbeddingRequest {
        texts: (0..100).map(|i| format!("Text {}", i)).collect(),
        content_type: ContentType::General,
        model_override: None,
        metadata: HashMap::new(),
    };

    let result = service.generate_batch(large_batch).await;
    assert!(result.is_err(), "Should fail with batch size exceeded");

    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("Batch size"),
        "Error should mention batch size"
    );
    println!("✅ Batch size validation works: {}", error_msg);
}

// Helper test to verify ishtar connectivity before running other tests
#[tokio::test]
async fn test_ishtar_connectivity_prerequisite() {
    // TDD: Basic connectivity test that other tests depend on (real Ollama or mock)
    let (config, _mock_server) = create_test_config().await;
    
    println!("Testing basic HTTP connectivity to {}...", config.base.service_url);
    
    let client = reqwest::Client::new();
    let url = format!("{}/api/tags", config.base.service_url);

    match client.get(&url).send().await {
        Ok(response) => {
            println!("✅ HTTP connection to {} successful", config.base.service_url);
            println!("   Status: {}", response.status());

            if response.status().is_success() {
                if let Ok(body) = response.text().await {
                    println!(
                        "   Response preview: {}",
                        if body.len() > 100 {
                            &body[..100]
                        } else {
                            &body
                        }
                    );
                }
            }
        }
        Err(e) => {
            panic!(
                "❌ Cannot connect to {}: {}\nService should be available (real or mock)",
                config.base.service_url, e
            );
        }
    }
}
