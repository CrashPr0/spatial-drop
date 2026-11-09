/** One request at a time. A failed worker is replaced on the next request. */
export class PhotoPreparation<Input extends object, Output extends { id: number }> {
  private worker: Worker | null = null;
  private sequence = 0;
  private finish: ((value: Output | null) => void) | null = null;
  constructor(private createWorker: () => Worker, private timeoutMs = 15000) {}

  cancel() {
    this.worker?.terminate();
    this.worker = null;
    this.finish?.(null);
  }

  request(input: Input): Promise<Output | null> {
    if (this.finish) throw new Error("Photo preparation requests must be serialized");
    return new Promise(resolve => {
      const id = ++this.sequence;
      const timer = setTimeout(() => this.cancel(), this.timeoutMs);
      this.finish = value => {
        clearTimeout(timer);
        this.finish = null;
        resolve(value);
      };
      try {
        const worker = this.worker ??= this.createWorker();
        // Ignore already queued messages/errors from a terminated worker.
        const current = () => this.worker === worker && this.sequence === id && !!this.finish;
        worker.onmessage = ({ data }: MessageEvent<Output>) => {
          if (current() && data.id === id) this.finish?.(data);
        };
        worker.onerror = worker.onmessageerror = () => { if (current()) this.cancel(); };
        worker.postMessage({ ...input, id });
      } catch {
        this.cancel();
      }
    });
  }
}
