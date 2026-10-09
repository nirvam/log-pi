.PHONY: all build release test lint fmt clean run

CARGO ?= cargo

all: build

build:
	$(CARGO) build

release:
	$(CARGO) build --release

test:
	$(CARGO) test -- --nocapture

lint:
	$(CARGO) clippy --all-targets -- -D warnings

fmt:
	$(CARGO) fmt --all -- --check

fmt-fix:
	$(CARGO) fmt --all

clean:
	$(CARGO) clean

run:
	$(CARGO) run --
