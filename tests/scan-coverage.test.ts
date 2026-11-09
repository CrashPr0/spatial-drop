import assert from "node:assert/strict";
import test from "node:test";
import { addPhotoToCoverage, SCANNED_FILL, coverageFraction, createCoverage, missingSides, nearestGap, guideGap, PoseTracker, segmentFill, turn, viewDirection, wrap } from "../pages-src/scan-coverage";

test("the sensor gives the compass heading and tilt of the back camera", () => {
  // phone upright (beta 90), turned 90° counterclockwise: the back camera faces west
  const west = viewDirection({ alpha: 90, beta: 90, gamma: 0 });
  assert.ok(Math.abs(west.heading - 270) < 1e-9, `heading ${west.heading}`);
  assert.ok(Math.abs(west.elevation) < 1e-9);
  // flat on its back, the camera looks straight up
  assert.ok(Math.abs(viewDirection({ alpha: 0, beta: 0, gamma: 0 }).elevation - -90) < 1e-9);
  assert.equal(Math.round(wrap(-30)), 330);
  assert.equal(turn(350, 10), 20);
  assert.equal(turn(10, 350), -20);
});

test("a photo fills only the segment it was taken in", () => {
  const coverage = createCoverage();
  addPhotoToCoverage(coverage, 0, 1);
  const fill = segmentFill(coverage);
  assert.ok(fill[0] > 0.7, `${fill[0]}`);
  assert.equal(fill.slice(2, 11).reduce((a, b) => a + b, 0), 0);
  assert.ok(coverageFraction(coverage) < 0.1);
});

test("linked photos fill the arc between them; unlinked ones leave a hole", () => {
  const linked = createCoverage();
  addPhotoToCoverage(linked, 0, 1);
  addPhotoToCoverage(linked, 40, 0.3);
  assert.ok(segmentFill(linked)[1] > 0.95);

  const jumped = createCoverage();
  addPhotoToCoverage(jumped, 0, 1);
  // 40° away but the views share nothing: the camera lost track in between
  addPhotoToCoverage(jumped, 40, 0.02);
  assert.equal(jumped.breaks, 1);
  assert.ok(segmentFill(jumped)[1] < SCANNED_FILL);
  // a huge sensor jump is drift, not travel, even with overlap
  const drift = createCoverage();
  addPhotoToCoverage(drift, 0, 1);
  addPhotoToCoverage(drift, 180, 0.5);
  assert.ok(segmentFill(drift)[3] === 0 && drift.breaks === 1);
});

test("a full lap closes the ring and the gaps are named", () => {
  const coverage = createCoverage();
  for (let at = 0; at < 360; at += 20) addPhotoToCoverage(coverage, at, 0.3);
  assert.equal(coverageFraction(coverage), 1);
  assert.equal(nearestGap(coverage, 0), null);
  assert.deepEqual(missingSides(coverage), []);

  const half = createCoverage();
  for (let at = 0; at <= 160; at += 20) addPhotoToCoverage(half, at, 0.3);
  assert.ok(coverageFraction(half) > 0.4 && coverageFraction(half) < 0.6);
  assert.ok(missingSides(half).includes("right"));
  assert.ok(!missingSides(half).includes("front"));
});

test("the arrow heads for the nearest open segment, ties going the way you are travelling", () => {
  const coverage = createCoverage();
  addPhotoToCoverage(coverage, 0, 1);
  assert.equal(nearestGap(coverage, 0, "ccw")?.direction, "ccw");
  assert.equal(nearestGap(coverage, 0, "cw")?.direction, "cw");
  // standing at 60° with 30°–0° scanned behind and nothing ahead
  for (let at = 20; at <= 60; at += 20) addPhotoToCoverage(coverage, at, 0.3);
  assert.equal(nearestGap(coverage, 60)?.direction, "cw");
});

test("the arrow keeps the travel direction past a half-filled segment's middle", () => {
  const coverage = createCoverage();
  for (let at = 0; at <= 50; at += 10) addPhotoToCoverage(coverage, at, 0.3);
  // at 62°, segment 2 (centred on 60°, 45–75°) is under its fill mark and its
  // middle is just behind: a nearest pick would point back, then forward again
  assert.ok(segmentFill(coverage)[2] < SCANNED_FILL);
  assert.equal(nearestGap(coverage, 62, "cw")?.direction, "cw");
  assert.equal(nearestGap(coverage, 58, "cw")?.direction, "cw");
  // walking the other way, it keeps pointing that way
  assert.equal(nearestGap(coverage, 62, "ccw")?.direction, "ccw");
});

test("tracking follows the sensor heading relative to where the scan began", () => {
  const pose = new PoseTracker();
  assert.equal(pose.angle(), null);
  pose.sense({ alpha: 40, beta: 90, gamma: 0 });
  pose.anchor();
  pose.sense({ alpha: 10, beta: 90, gamma: 0 });
  // walking left round the object: the camera turns right, alpha falls, the heading rises
  assert.equal(Math.round(pose.angle()!), 30);
  pose.sense({ alpha: 70, beta: 90, gamma: 0 });
  assert.equal(Math.round(pose.angle()!), 330);
});

test("without a sensor the angle is estimated from the image shift", () => {
  const pose = new PoseTracker();
  assert.equal(pose.angleForKeep(null, 0, 1), 0);
  // tiles slid right: the camera went right, the angle drops
  assert.ok(pose.angleForKeep(0, 0.1, 0.3) > 300);
  assert.ok(pose.angleForKeep(0, -0.1, 0.3) < 60);
});


test("gap guidance keeps its destination until filled and reports real angular distance", () => {
  const coverage = createCoverage();
  coverage.seen.fill(1);
  for (let i = 15; i < 45; i++) coverage.seen[i] = 0;
  for (let i = 75; i < 105; i++) coverage.seen[i] = 0;
  const first = guideGap(coverage, 50, null, "ccw")!;
  assert.equal(first.segment, 1);
  assert.equal(first.degrees, 20);
  const retained = guideGap(coverage, 70, first.segment, "cw")!;
  assert.equal(retained.segment, 1);
  assert.equal(retained.degrees, 40);
  addPhotoToCoverage(coverage, 30, 0);
  assert.equal(guideGap(coverage, 30, first.segment)?.segment, 3);
  coverage.seen.fill(1);
  assert.equal(guideGap(coverage, 30, 3), null);
});

test("gap distance excludes direction preference penalty across zero", () => {
  const coverage = createCoverage();
  coverage.seen.fill(1);
  for (let i = 0; i < 15; i++) coverage.seen[i] = 0;
  for (let i = 345; i < 360; i++) coverage.seen[i] = 0;
  assert.equal(nearestGap(coverage, 350, "ccw")?.degrees, 10);
});
