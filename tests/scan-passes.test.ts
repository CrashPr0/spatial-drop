import assert from "node:assert/strict";
import test from "node:test";
import { PASSES, passDone, passProgress, RING_DONE, tiltAdvice } from "../pages-src/scan-passes";

const [low, high, top] = PASSES;

test("the steps go around, then from above, then over the top", () => {
  assert.deepEqual(PASSES.map((pass) => pass.id), ["low", "high", "top"]);
  // every step's band sits lower than the one before
  for (let i = 1; i < PASSES.length; i++) assert.ok(PASSES[i].maxElevation <= PASSES[i - 1].maxElevation);
  // the budgets fit the photo limit
  assert.ok(PASSES.reduce((sum, pass) => sum + pass.budget, 0) <= 50);
});

test("tilt advice: level for the first loop, pointed down for the second", () => {
  assert.equal(tiltAdvice(low, -5), "ok");
  assert.equal(tiltAdvice(low, -50), "lower");
  assert.equal(tiltAdvice(high, -5), "raise");
  assert.equal(tiltAdvice(high, -40), "ok");
  assert.equal(tiltAdvice(top, -30), "raise");
  assert.equal(tiltAdvice(top, -85), "ok");
  // the climb over the top picks up right where the loop from above ends
  assert.ok(top.continues && top.maxElevation >= high.minElevation);
  // a wobble just past the band edge is let through
  assert.equal(tiltAdvice(high, -22), "ok");
  assert.equal(tiltAdvice(high, null), "unknown");
});

test("a loop is done when the ring is covered or the step's photos run out", () => {
  assert.equal(passProgress(low, RING_DONE / 2, 10), 0.5);
  assert.equal(passDone(low, 0.5, 10), false);
  assert.equal(passDone(low, RING_DONE, 20), true);
  assert.equal(passDone(low, 0.4, low.budget), true);
  assert.equal(passProgress(top, 0, 2), 2 / 4);
  assert.equal(passDone(top, 0, 4), true);
});
