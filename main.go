package main

import (
	"fmt"
	"os"

	"github.com/vingarcia/kortex/internal/proxy"
)

func main() {
	selfPath, err := os.Executable()
	if err != nil {
		selfPath = os.Args[0]
	}
	claudeBin, err := proxy.ResolveClaudeBin(os.Getenv("KORTEX_CLAUDE_BIN"), selfPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "kortex: %v\n", err)
		os.Exit(1)
	}
	os.Exit(proxy.Run(proxy.Config{
		ClaudeBin: claudeBin,
		Argv:      os.Args[1:],
		LogPath:   os.Getenv("KORTEX_LOG"),
		Stdin:     os.Stdin,
		Stdout:    os.Stdout,
		Stderr:    os.Stderr,
	}))
}
