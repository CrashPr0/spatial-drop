export interface CapturePose { step: string; ringAngle: number; elevation: number | null }
/** Compare relative viewing-direction changes; no assumption about world heading.
 * Compass resets across passes and large sensor jumps are deliberately excluded.
 */
export function poseDisagreement(views: { R: ArrayLike<number>; note?: CapturePose }[]) {
  const errors: number[] = [];
  const rad = Math.PI / 180;
  for (let i = 1; i < views.length; i++) {
    const a = views[i - 1], b = views[i], na = a.note, nb = b.note;
    if (!na || !nb || na.step !== nb.step || na.elevation === null || nb.elevation === null) continue;
    const yaw = ((nb.ringAngle - na.ringAngle + 540) % 360) - 180;
    if (Math.abs(yaw) > 30) continue;
    const cosine = Math.sin(na.elevation * rad) * Math.sin(nb.elevation * rad)
      + Math.cos(na.elevation * rad) * Math.cos(nb.elevation * rad) * Math.cos(yaw * rad);
    const sensor = Math.acos(Math.max(-1, Math.min(1, cosine))) / rad;
    const dot = a.R[6] * b.R[6] + a.R[7] * b.R[7] + a.R[8] * b.R[8];
    const solved = Math.acos(Math.max(-1, Math.min(1, dot))) / rad;
    if (Number.isFinite(sensor) && Number.isFinite(solved)) errors.push(Math.abs(sensor - solved));
  }
  errors.sort((a, b) => a - b);
  const median = errors.length ? errors[errors.length >> 1] : 0;
  return { pairs: errors.length, medianDegrees: median, warning: errors.length >= 6 && median > 15 };
}
