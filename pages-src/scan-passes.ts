// A guided capture in steps: one loop at the object's height, one from above
// looking down, then a few photos straight over the top. Each step owns a
// band of camera tilt (the elevation of the view, from the phone's sensors):
// photos are only kept inside it, so the person cannot drift into a single
// muddled loop. Without a sensor the tilt is unknown and the steps are only
// instructions.

export type PassId = "low" | "high" | "top";

export interface ScanPass {
  id: PassId;
  title: string;
  /** one sentence on what to do, shown under the steps */
  how: string;
  /** view elevation band in degrees, negative looking down */
  minElevation: number;
  maxElevation: number;
  /** photos kept before the step ends regardless of coverage */
  budget: number;
  /** photos kept back for this step while the steps before it run */
  reserve: number;
  /** "ring": done when the loop is covered; "count": done after `count` photos */
  goal: "ring" | "count";
  count?: number;
  /**
   * Carry on the photo chain from the step before instead of starting a new
   * one. Over the top, photos straight down taken after a jump from the loop
   * share too little with it: on the synthetic orbit none of them could be
   * placed. Rising in small steps from the last loop photo keeps them linked.
   */
  continues?: boolean;
}

export const PASSES: ScanPass[] = [
  {
    id: "low", title: "Around", minElevation: -25, maxElevation: 25, budget: 30, reserve: 0, goal: "ring",
    how: "Hold the phone level with the middle of the object and walk all the way around it.",
  },
  {
    id: "high", title: "From above", minElevation: -65, maxElevation: -25, budget: 28, reserve: 45, goal: "ring",
    how: "Raise the phone above the object, tilt it down at the object, and walk around it again.",
  },
  {
    id: "top", title: "Over the top", minElevation: -90, maxElevation: -45, budget: 5, reserve: 5, goal: "count", count: 5, continues: true,
    how: "From where you are, move the phone up and over the object, tilting it down, and hold still at each stop.",
  },
];

/** share of a loop that counts as all the way around (11 of 12 ring segments) */
export const RING_DONE = 11 / 12;

export type Tilt = "ok" | "raise" | "lower" | "unknown";

/**
 * Whether the camera's tilt suits the step. "raise" means the view is too
 * level for it: lift the phone and point it down at the object. `slack`
 * widens the band so a hand wobbling on its edge is not nagged.
 */
export function tiltAdvice(pass: ScanPass, elevation: number | null, slack = 5): Tilt {
  if (elevation === null || !Number.isFinite(elevation)) return "unknown";
  if (elevation > pass.maxElevation + slack) return "raise";
  if (elevation < pass.minElevation - slack) return "lower";
  return "ok";
}

/** 0–1 progress through a step, from its loop coverage or its photo count. */
export function passProgress(pass: ScanPass, coverage: number, photos: number) {
  const progress = pass.goal === "ring" ? coverage / RING_DONE : photos / (pass.count ?? pass.budget);
  return Math.max(0, Math.min(1, progress));
}

// With real orientation readings, a photo count is not evidence of a full
// orbit. Keep the count fallback for devices without usable sensors.
export const passDone = (pass: ScanPass, coverage: number, photos: number, tracked = false) =>
  passProgress(pass, coverage, photos) >= 1 || ((!tracked || pass.goal !== "ring") && photos >= pass.budget);


/** How many photos the scan may hold by the end of a step: the rest belong to the steps after it. */
export const stepCeiling = (index: number, maxPhotos: number) =>
  maxPhotos - PASSES.slice(index + 1).reduce((sum, pass) => sum + pass.reserve, 0);

/** Conservative acceptance gate; photo count alone does not establish coverage. */
export const hasEnoughSolvedViews = (placed: number, total: number) =>
  total >= 12 && placed >= Math.max(12, Math.ceil(total * 0.7));
