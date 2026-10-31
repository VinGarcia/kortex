package tools

import "time"

// StartKeepalive runs a goroutine that calls emit every interval until the
// returned stop func is invoked. It is the shared primitive behind every
// keepalive span kortex emits to outpace OpenClaw's no-output watchdog (180s):
// the governor seam brackets the superego ladder with it (see
// session.startGovernorKeepalive), and RunLoop brackets each silent core-model
// call with it (see StreamEmitter.startCallKeepalive).
//
// The returned stop func closes the done channel AND blocks until the goroutine
// has exited. Callers MUST invoke stop BEFORE emitting any terminal line
// (assistant / result / tool_result): the stream-json wire order must stay
// deterministic, so no keepalive may land after the terminal result. (The
// emitter serializes its own writes, so this is an ordering contract, not a
// data-race one — see StreamEmitter.writeMu.)
//
// A panic in the ticker goroutine (a broken writer, a future emitter bug) is
// recovered and routed to logf, never allowed to reach the runtime: this
// goroutine has no caller frame to recover it, so an unrecovered panic would
// kill the whole CLI mid-turn — the exact outage the keepalive exists to
// prevent. logf may be nil when the caller has no diagnostic sink (the tools
// package carries no logger); the panic is still recovered, only not reported.
func StartKeepalive(interval time.Duration, emit func() error, logf func(format string, args ...any)) (stop func()) {
	done := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		defer func() {
			if r := recover(); r != nil && logf != nil {
				logf("keepalive goroutine panic: %v", r)
			}
		}()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if err := emit(); err != nil {
					if logf != nil {
						logf("keepalive emit failed: %v", err)
					}
					return
				}
			}
		}
	}()
	return func() {
		close(done)
		<-finished
	}
}
