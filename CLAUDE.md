# CLAUDE.md - tyl-ollama-embedding-adapter

## 📋 **Module Context**

**tyl-ollama-embedding-adapter** is the Ollama embedding adapter module for the TYL framework with batch processing and content-type optimization.

## 🏗️ **Architecture**

### **Port (Interface)**
```rust
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn generate_embedding(&self, text: &str, content_type: ContentType) -> EmbeddingResult<Embedding>;
    async fn generate_batch(&self, request: BatchEmbeddingRequest) -> EmbeddingResult<BatchEmbeddingResponse>;
    fn get_model_dimensions(&self, model: &str) -> EmbeddingResult<usize>;
    async fn health_check(&self) -> EmbeddingResult<HealthCheckResult>;
    fn supported_models(&self) -> Vec<String>;
    fn config(&self) -> &EmbeddingConfig;
}
```

### **Adapters (Implementations)**
- `OllamaEmbeddingService` - Ollama-specific implementation with content-type optimization and batch processing

### **Core Types**
- `OllamaConfig` - Ollama-specific configuration extending EmbeddingConfig
- `EmbeddingResult<T>` - Result type alias using TYL error handling
- `ContentType` - Content type enumeration for model optimization

## 🧪 **Testing**

```bash
cargo test -p tyl-ollama-embedding-adapter
cargo test --doc -p tyl-ollama-embedding-adapter
cargo run --example basic_usage -p tyl-ollama-embedding-adapter
```

## 📂 **File Structure**

```
tyl-ollama-embedding-adapter/
├── src/lib.rs                 # Core Ollama adapter implementation
├── examples/
│   └── basic_usage.rs         # Comprehensive usage examples
├── tests/
│   └── integration_tests.rs   # Integration tests
├── README.md                  # Main documentation
├── CLAUDE.md                  # This file
└── Cargo.toml                 # Package metadata
```

## 🔧 **How to Use**

### **Basic Usage**
```rust
use tyl_ollama_embedding_adapter::{OllamaEmbeddingService, OllamaConfig};
use tyl_embeddings_port::{EmbeddingService, ContentType};

// Create Ollama embedding service
let config = OllamaConfig::default();
let service = OllamaEmbeddingService::from_config(config).await?;

// Generate embeddings with content-type optimization
let code_embedding = service.generate_embedding(
    "fn main() { println!(\"Hello, world!\"); }", 
    ContentType::Code
).await?;

let text_embedding = service.generate_embedding(
    "This is a documentation example.", 
    ContentType::Documentation
).await?;
```

### **Batch Processing**
```rust
use tyl_embeddings_port::BatchEmbeddingRequest;
use std::collections::HashMap;

let batch_request = BatchEmbeddingRequest {
    texts: vec![
        "First document".to_string(),
        "Second document".to_string(),
        "Third document".to_string(),
    ],
    content_type: ContentType::Documentation,
    model_override: None,
    metadata: HashMap::new(),
};

let batch_response = service.generate_batch(batch_request).await?;
println!("Generated {} embeddings", batch_response.embeddings.len());
```

### **Custom Configuration**
```rust
let mut config = OllamaConfig::default();
config.host = "remote-ollama-server".to_string();
config.port = 11434;
config.auto_pull_models = false;
config.connection_pool_size = 20;

// Override model mapping for specific content types
config.model_mapping.insert("code".to_string(), "custom-code-model:latest".to_string());
config.model_mapping.insert("text".to_string(), "custom-text-model:latest".to_string());

let service = OllamaEmbeddingService::from_config(config).await?;
```

## 🛠️ **Useful Commands**

```bash
cargo clippy -p tyl-{module-name}
cargo fmt -p tyl-{module-name}  
cargo doc --no-deps -p tyl-{module-name} --open
cargo test -p tyl-{module-name} --verbose
```

## 📦 **Dependencies**

### **Runtime**
- `serde` - Serialization support
- `serde_json` - JSON handling
- `thiserror` - Error handling
- `uuid` - Unique identifier generation

### **Development**
- Standard Rust testing framework

## 🎯 **Design Principles**

1. **Hexagonal Architecture** - Clean separation of concerns
2. **Trait-based Extensibility** - Easy to add new implementations
3. **Error Handling** - Comprehensive error types with context
4. **Serialization** - First-class serde support
5. **Testing** - Comprehensive test coverage

## ⚠️ **Known Limitations**

- {Add any current limitations}
- {Add any planned improvements}

## 📝 **Notes for Contributors**

- Follow TDD approach
- Maintain hexagonal architecture
- Document all public APIs with examples
- Add integration tests for new features
- Keep dependencies minimal

## 🔗 **Related TYL Modules**

- [`tyl-errors`](https://github.com/the-yaml-life/tyl-errors) - Error handling
- [`tyl-logging`](https://github.com/the-yaml-life/tyl-logging) - Structured logging