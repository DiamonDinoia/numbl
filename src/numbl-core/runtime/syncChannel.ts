/**
 * Synchronous communication channel using SharedArrayBuffer + Atomics.
 *
 * Allows a worker thread to synchronously block until the main thread
 * responds. Used for operations like pause() and input() that need
 * to wait without async/await.
 *
 * Protocol:
 *   1. Worker posts a message to main thread with request details
 *   2. Worker calls Atomics.wait(ctrl, 0, 0) — blocks until signaled
 *   3. Main thread processes request, writes response to data buffer
 *   4. Main thread calls Atomics.store(ctrl, 0, 1) + Atomics.notify(ctrl, 0)
 *   5. Worker wakes up, reads response, resets control word to 0
 */

// Control word values
const CTRL_IDLE = 0;
const CTRL_RESPONSE_READY = 1;

/**
 * Synchronous sleep using Atomics.wait with a timeout.
 * Works in Web Workers and Node.js (including main thread).
 * No SharedArrayBuffer communication needed — just a local buffer.
 */
let syncSleepWarned = false;

export function syncSleep(ms: number): void {
  if (ms <= 0) return;
  if (typeof SharedArrayBuffer !== "undefined") {
    const sab = new SharedArrayBuffer(4);
    const int32 = new Int32Array(sab);
    Atomics.wait(int32, 0, 0, ms);
  } else {
    // Fallback: busy-wait when SharedArrayBuffer is unavailable
    // (e.g., browser without Cross-Origin Isolation headers)
    if (!syncSleepWarned) {
      syncSleepWarned = true;
      console.warn(
        "SharedArrayBuffer is not available — pause() will busy-wait. " +
          "Enable Cross-Origin Isolation headers (COOP/COEP) for efficient blocking."
      );
    }
    const end = Date.now() + ms;
    while (Date.now() < end) {
      /* spin */
    }
  }
}

/** Buffers shared between worker and main thread for sync communication. */
export interface SyncChannelBuffers {
  controlBuffer: SharedArrayBuffer;
  dataBuffer: SharedArrayBuffer;
}

/** Create the shared buffers for a sync channel. Call on main thread. */
export function createSyncChannelBuffers(
  dataBufferSize = 65536
): SyncChannelBuffers {
  return {
    controlBuffer: new SharedArrayBuffer(4),
    dataBuffer: new SharedArrayBuffer(dataBufferSize),
  };
}

/**
 * Worker-side sync channel. Sends requests and blocks until response.
 * The `postMessage` callback should send the request to the main thread.
 */
export class SyncChannel {
  private ctrl: Int32Array;
  private data: Uint8Array;
  private postMessage: (msg: unknown) => void;

  constructor(
    buffers: SyncChannelBuffers,
    postMessage: (msg: unknown) => void
  ) {
    this.ctrl = new Int32Array(buffers.controlBuffer);
    this.data = new Uint8Array(buffers.dataBuffer);
    this.postMessage = postMessage;
  }

  /**
   * Send a request to the main thread and block until response.
   * Returns the response as a string (decoded from the data buffer).
   */
  request(type: string, payload?: Record<string, unknown>): string {
    // Reset control word
    Atomics.store(this.ctrl, 0, CTRL_IDLE);

    // Post the request to main thread
    this.postMessage({ type: "sync_request", request: { type, ...payload } });

    // Block until main thread signals response
    Atomics.wait(this.ctrl, 0, CTRL_IDLE);

    // Read response length from first 4 bytes of data buffer
    const view = new DataView(this.data.buffer);
    const len = view.getUint32(0, true);

    // Decode response string
    const responseBytes = this.data.slice(4, 4 + len);
    const response = new TextDecoder().decode(responseBytes);

    // Reset control word for next use
    Atomics.store(this.ctrl, 0, CTRL_IDLE);

    return response;
  }
}

/**
 * Main-thread side: respond to a sync channel request.
 * Call this from the main thread's message handler when receiving
 * a { type: "sync_request" } message.
 */
export function respondToSyncRequest(
  buffers: SyncChannelBuffers,
  response: string
): void {
  const data = new Uint8Array(buffers.dataBuffer);
  const view = new DataView(buffers.dataBuffer);
  const encoded = new TextEncoder().encode(response);

  // Write response length + data
  view.setUint32(0, encoded.length, true);
  data.set(encoded, 4);

  // Signal the worker
  const ctrl = new Int32Array(buffers.controlBuffer);
  Atomics.store(ctrl, 0, CTRL_RESPONSE_READY);
  Atomics.notify(ctrl, 0);
}
