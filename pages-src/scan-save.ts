/** Serialize backups while retaining only the newest waiting snapshot. */
export class LatestScanSave<T> {
  private pending: { value: T } | null = null;
  private running = false;
  constructor(private write: (value: T) => Promise<unknown>, private failed: () => void) {}
  save(value: T) {
    this.pending = { value };
    if (!this.running) void this.drain();
  }
  private async drain() {
    this.running = true;
    try {
      while (this.pending) {
        const { value } = this.pending;
        this.pending = null;
        try { await this.write(value); } catch { this.failed(); }
      }
    } finally { this.running = false; }
  }
}
