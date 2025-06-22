# Image de base avec Rust et dépendances C++
FROM rust:1.84-bookworm

# Installer les dépendances système pour XGBoost
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libclang-dev \
    clang \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Créer le répertoire de travail
WORKDIR /app

# Copier les fichiers de configuration Rust
COPY Cargo.toml ./

# Copier le code source
COPY src/ ./src/
COPY train.csv test.csv ./

# Construire l'application
RUN cargo build --release

# Point d'entrée
CMD ["./target/release/regression"] 