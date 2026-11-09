/** Keep decoded previews across captures and release URLs even if loading fails. */
export function photoThumbnails(container: HTMLElement) {
  const previews = new Map<File, { image: HTMLImageElement; release: () => void }>();
  return (photos: readonly File[]) => {
    const visible = photos.slice(-12);
    const wanted = new Set(visible);
    for (const [file, preview] of previews) {
      if (wanted.has(file)) continue;
      preview.release();
      preview.image.removeAttribute("src");
      preview.image.remove();
      previews.delete(file);
    }
    const images = visible.map(file => {
      let preview = previews.get(file);
      if (!preview) {
        const image = container.ownerDocument.createElement("img");
        const url = URL.createObjectURL(file);
        let released = false;
        const release = () => {
          if (!released) URL.revokeObjectURL(url);
          released = true;
        };
        image.alt = "";
        image.decoding = "async";
        image.addEventListener("load", release, { once: true });
        image.addEventListener("error", release, { once: true });
        image.src = url;
        preview = { image, release };
        previews.set(file, preview);
      }
      return preview.image;
    });
    // Ordinary capture only appends one new node; unchanged previews stay put.
    images.forEach((image, index) => {
      if (container.children[index] !== image) container.insertBefore(image, container.children[index] ?? null);
    });
  };
}
