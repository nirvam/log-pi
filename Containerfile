# Stage 1: Build binary
FROM rust:1.80-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -U logpi

COPY --from=builder /app/target/release/log-pi /usr/local/bin/log-pi

USER logpi
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/log-pi"]
CMD ["--help"]
