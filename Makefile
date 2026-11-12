GO_ARCH_LINT ?= $(shell command -v go-arch-lint 2>/dev/null || echo $(HOME)/go/bin/go-arch-lint)

lint:
	@unformatted=$$(gofmt -l .); if [ -n "$$unformatted" ]; then echo "gofmt needed:"; echo "$$unformatted"; exit 1; fi
	go vet ./...
	$(GO_ARCH_LINT) check

test:
	go test ./...

smoke:
	./hack/smoke.sh

.PHONY: lint test smoke install

install: test
	GOBIN=$(HOME)/.local/bin go install .
	GOBIN=$(HOME)/go/bin go install .
