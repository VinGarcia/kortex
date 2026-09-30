GO_ARCH_LINT ?= $(shell command -v go-arch-lint 2>/dev/null || echo $(HOME)/go/bin/go-arch-lint)

lint:
	go vet ./...
	$(GO_ARCH_LINT) check

test:
	go test ./...

smoke:
	./hack/smoke.sh

.PHONY: lint test smoke
