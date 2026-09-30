package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/vingarcia/kortex/internal/proxy"
)

// Env vars OpenClaw uses to announce the FD carrying the auth secret
// (spec F0 §2). Only one is set per spawn; both are honored if present.
var authFDEnvVars = []string{
	"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
	"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
}

func main() {
	selfPath, err := os.Executable()
	if err != nil {
		selfPath = os.Args[0]
	}
	searchDirs := filepath.SplitList(os.Getenv("PATH"))
	claudeBin, err := proxy.ResolveClaudeBin(os.Getenv("KORTEX_CLAUDE_BIN"), selfPath, searchDirs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "kortex: %v\n", err)
		os.Exit(1)
	}
	os.Exit(proxy.Run(proxy.Config{
		ClaudeBin: claudeBin,
		Argv:      os.Args[1:],
		AuthFDs:   authFDsFromEnv(),
		LogPath:   os.Getenv("KORTEX_LOG"),
		Stdin:     os.Stdin,
		Stdout:    os.Stdout,
		Stderr:    os.Stderr,
	}))
}

func authFDsFromEnv() []int {
	var fds []int
	for _, name := range authFDEnvVars {
		value := os.Getenv(name)
		if value == "" {
			continue
		}
		fd, err := strconv.Atoi(value)
		if err != nil {
			fmt.Fprintf(os.Stderr, "kortex: auth fd not forwarded: %s=%q is not a number\n", name, value)
			continue
		}
		fds = append(fds, fd)
	}
	return fds
}
