# TYL Ollama Embedding Adapter

🧠 **Ollama embedding adapter for TYL framework with batch processing and content-type optimization**

This module provides a complete Ollama integration for the TYL framework's embedding port, enabling local embedding generation with automatic model optimization and efficient batch processing.

[![Crates.io](https://img.shields.io/crates/v/tyl-ollama-embedding-adapter.svg)](https://crates.io/crates/tyl-ollama-embedding-adapter)
[![Documentation](https://docs.rs/tyl-ollama-embedding-adapter/badge.svg)](https://docs.rs/tyl-ollama-embedding-adapter)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)

## ✨ Features

- **🚀 Batch Processing**: Automatic chunking for large requests with dimension validation
- **🎯 Content-Type Optimization**: Different models for code vs. text using ContentType enum
- **🔧 Ollama Integration**: Native support for Ollama embedding API with auto model pulling
- **📊 Health Monitoring**: Built-in health checks and model availability verification
- **⚙️ TYL Framework Integration**: Full integration with TYL error handling, config, logging, and tracing
- **🏗️ Hexagonal Architecture**: Clean separation between port interface and Ollama adapter

## 🚀 Quick Start

### Add to Your Project

```toml
[dependencies]
tyl-ollama-embedding-adapter = { git = "https://github.com/the-yaml-life/tyl-ollama-embedding-adapter.git" }
tyl-embeddings-port = { git = "https://github.com/the-yaml-life/tyl-embeddings-port.git" }
tokio = { version = "1.0", features = ["full"] }
```

### Basic Usage

```rust
use tyl_ollama_embedding_adapter::{OllamaEmbeddingService, OllamaConfig};
use tyl_embeddings_port::{EmbeddingService, ContentType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create Ollama embedding service with default configuration
    let config = OllamaConfig::default(); // Connects to localhost:11434
    let service = OllamaEmbeddingService::from_config(config).await?;

    // Generate embeddings with content-type optimization
    let code_embedding = service.generate_embedding(
        "fn main() { println!(\"Hello, world!\"); }", 
        ContentType::Code  // Automatically uses optimized model for code
    ).await?;

    let text_embedding = service.generate_embedding(
        "This is a documentation example.", 
        ContentType::Documentation  // Uses text-optimized model
    ).await?;

    println!("Code embedding: {} dimensions", code_embedding.dimensions());
    println!("Text embedding: {} dimensions", text_embedding.dimensions());

    Ok(())
}
```

### Batch Processing

```rust
use tyl_embeddings_port::BatchEmbeddingRequest;
use std::collections::HashMap;

// Process multiple texts efficiently
let batch_request = BatchEmbeddingRequest {
    texts: vec![
        "First document to embed".to_string(),
        "Second document for processing".to_string(),
        "Third piece of content".to_string(),
    ],
    content_type: ContentType::Documentation,
    model_override: None, // Uses content-type optimized model
    metadata: HashMap::new(),
};

let batch_response = service.generate_batch(batch_request).await?;
println!("Generated {} embeddings", batch_response.embeddings.len());
```

## 🔧 Configuration

### Environment Variables

Configure Ollama connection and behavior:

```bash
# Ollama server configuration
export TYL_OLLAMA_HOST=localhost
export TYL_OLLAMA_PORT=11434

# Model configuration
export TYL_OLLAMA_CODE_MODEL=codellama:7b
export TYL_OLLAMA_TEXT_MODEL=nomic-embed-text:latest
export TYL_OLLAMA_AVAILABLE_MODELS=nomic-embed-text:latest,codellama:7b

# Performance settings
export TYL_OLLAMA_CONNECTION_POOL_SIZE=10
export TYL_OLLAMA_RETRY_ATTEMPTS=3
export TYL_OLLAMA_AUTO_PULL_MODELS=true
```

### Custom Configuration

```rust
let mut config = OllamaConfig::default();

// Configure remote Ollama server
config.host = "remote-ollama-server".to_string();
config.port = 11434;

// Customize model mapping
config.model_mapping.insert("code".to_string(), "custom-code-model:latest".to_string());
config.model_mapping.insert("text".to_string(), "custom-text-model:latest".to_string());

// Performance tuning
config.connection_pool_size = 20;
config.auto_pull_models = false; // Disable auto-pulling in production

let service = OllamaEmbeddingService::from_config(config).await?;
```

## 📊 Content-Type Optimization

The adapter automatically selects optimal models based on content type:

| Content Type | Default Model | Use Case |
|--------------|---------------|----------|
| `Code` | `codellama:7b` | Source code, technical content |
| `Documentation` | `nomic-embed-text:latest` | User docs, explanations |
| `Query` | `nomic-embed-text:latest` | Search terms, user queries |
| `Task` | `codellama:7b` | Task descriptions, requirements |
| `Memory` | `nomic-embed-text:latest` | Memory content, notes |
| `General` | `nomic-embed-text:latest` | Default text content |

## 🧪 Testing

### Unit Tests
```bash
cargo test --lib
```

### Integration Tests (Requires Ollama)
```bash
# Start Ollama server first
ollama serve

# Run integration tests
cargo test --test integration_tests
```

### Example Usage
```bash
cargo run --example basic_usage
```

## 🏗️ Architecture

This module follows hexagonal architecture:

- **Port (Interface)**: `EmbeddingService` from `tyl-embeddings-port`
- **Adapter**: `OllamaEmbeddingService` - Ollama-specific implementation
- **Configuration**: `OllamaConfig` with TYL framework integration
- **Domain Logic**: Content-type optimization and batch processing

```
┌─────────────────────────────────────────────┐
│              Application Layer              │
│  (Your code using EmbeddingService trait)   │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│           EmbeddingService Port             │
│        (trait from tyl-embeddings-port)     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         OllamaEmbeddingService              │
│              (this adapter)                 │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              Ollama API                     │
│         (localhost:11434/api/embeddings)    │
└─────────────────────────────────────────────┘
```

## 🚀 Performance

### Benchmarks (nomic-embed-text:latest on localhost)

- **Single embedding**: ~65ms average
- **Batch of 5 texts**: ~200ms average  
- **Dimensions**: 768 (nomic-embed-text)
- **Max batch size**: 50 (configurable)

### Optimization Tips

1. **Use batch processing** for multiple texts
2. **Configure connection pooling** for high throughput
3. **Pre-pull models** to avoid download delays
4. **Use content-type optimization** for better quality

## 🔗 Related TYL Modules

- [`tyl-embeddings-port`](https://github.com/the-yaml-life/tyl-embeddings-port) - Embedding port interface
- [`tyl-errors`](https://github.com/the-yaml-life/tyl-errors) - Error handling
- [`tyl-config`](https://github.com/the-yaml-life/tyl-config) - Configuration management
- [`tyl-logging`](https://github.com/the-yaml-life/tyl-logging) - Structured logging  
- [`tyl-tracing`](https://github.com/the-yaml-life/tyl-tracing) - Distributed tracing

## 📝 Requirements

- **Rust**: 1.70+ (2021 edition)
- **Ollama**: Latest version running locally or remotely
- **Models**: At least `nomic-embed-text:latest` for basic functionality

## 🤝 Contributing

See [CLAUDE.md](CLAUDE.md) for development context and patterns.

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD)
4. Implement the feature
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for the excellent local LLM platform
- [TYL Framework](https://github.com/the-yaml-life) for hexagonal architecture patterns
- The Rust community for amazing ecosystem tools