/** Best effort only; screen wake lock cannot guarantee GPU availability. */
export function scanWakeLock() {
  let wanted = false, requesting = false;
  let lock: WakeLockSentinel | null = null;
  const update = async (enabled: boolean) => {
    wanted = enabled;
    if (!enabled || document.hidden) { const old = lock; lock = null; await old?.release().catch(() => {}); return; }
    if (lock || requesting || !navigator.wakeLock) return;
    requesting = true;
    try {
      const next = await navigator.wakeLock.request("screen");
      if (!wanted || document.hidden) await next.release();
      else { lock = next; next.addEventListener("release", () => { if (lock === next) lock = null; }); }
    } catch { /* unsupported or denied: scanning still works */ }
    finally { requesting = false; }
  };
  document.addEventListener("visibilitychange", () => { void update(wanted); });
  window.addEventListener("pagehide", () => { void update(false); });
  return update;
}
