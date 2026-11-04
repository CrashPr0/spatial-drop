// A single local draft. Storage is optional: quota/private-mode failures must
// never stop reconstruction or imply that a draft was successfully saved.
export async function scanDraft<T>(action: "read" | "write" | "clear", value?: T): Promise<T | undefined> {
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open("spatial-drop-scan", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("draft");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("Scan storage is blocked"));
  });
  try {
    return await new Promise<T | undefined>((resolve, reject) => {
      const tx = db.transaction("draft", action === "read" ? "readonly" : "readwrite");
      const store = tx.objectStore("draft");
      const request = action === "read" ? store.get("current") : action === "clear" ? store.delete("current") : store.put(value, "current");
      tx.oncomplete = () => resolve(action === "read" ? request.result : undefined);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally { db.close(); }
}
