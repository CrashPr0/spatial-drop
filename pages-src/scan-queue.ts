/** Serial worker access with bounded outstanding candidates and cancellation. */
export class CaptureQueue<T> {
  private tasks: { value: T; resolve: (result: boolean | null) => void }[] = [];
  private generation = 0;
  constructor(private process: (value: T) => Promise<boolean | null>, private changed: () => void, readonly capacity = 3) {}
  get pending() { return this.tasks.length; }
  enqueue(value: T): Promise<boolean | null> | null {
    if (this.pending >= this.capacity) return null;
    const result = new Promise<boolean | null>((resolve) => this.tasks.push({value, resolve}));
    this.changed();
    if (this.pending === 1) void this.run(this.generation);
    return result;
  }
  reset() {
    this.generation++;
    const tasks = this.tasks;
    this.tasks = [];
    tasks.forEach(t => t.resolve(null));
    this.changed();
  }
  private async run(generation: number) {
    while (this.tasks.length && generation === this.generation) {
      const task = this.tasks[0];
      let result: boolean | null = null;
      try { result = await this.process(task.value); } catch { /* preserve unknown candidates */ }
      if (generation !== this.generation) return;
      this.tasks.shift();
      task.resolve(result);
      this.changed();
    }
  }
}
