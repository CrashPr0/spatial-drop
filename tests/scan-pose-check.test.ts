import assert from 'node:assert/strict';
import test from 'node:test';
import { poseDisagreement } from '../pages-src/scan-pose-check';
const views = (step: number) => Array.from({length: 8}, (_, i) => {
  const a = (i * step + 90) * Math.PI / 180;
  return {R: [Math.cos(a),0,-Math.sin(a),0,1,0,Math.sin(a),0,Math.cos(a)], note: {step:'low',ringAngle:i*10,elevation:0}};
});
test('relative pose check ignores world heading and detects repeated disagreement', () => {
  assert.equal(poseDisagreement(views(10)).warning, false);
  const bad = poseDisagreement(views(35));
  assert.equal(bad.warning, true);
  assert.ok(Math.abs(bad.medianDegrees - 25) < 1e-6);
});
test('missing sensors, pass resets and sparse evidence do not flag scans', () => {
  assert.equal(poseDisagreement(views(35).slice(0, 3)).warning, false);
  assert.equal(poseDisagreement(views(35).map(v => ({R:v.R}))).pairs, 0);
  assert.equal(poseDisagreement(views(35).map((v,i) => ({...v,note:{...v.note,step:String(i)}}))).pairs, 0);
});
