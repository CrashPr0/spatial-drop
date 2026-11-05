import assert from 'node:assert/strict';
import test from 'node:test';
import { scanWakeLock } from '../pages-src/scan-wake';
test('wake lock releases on hide, reacquires on return and releases when stopped', async () => {
  const descriptors = ['document','window','navigator'].map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)] as const);
  const events: Record<string,()=>void>={};
  let requests=0,releases=0;
  const document={hidden:false,addEventListener:(name:string,fn:()=>void)=>{events[name]=fn;}};
  Object.defineProperty(globalThis,'document',{configurable:true,value:document});
  Object.defineProperty(globalThis,'window',{configurable:true,value:{addEventListener:()=>{}}});
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{wakeLock:{request:async()=>{
    requests++; return {release:async()=>{releases++;},addEventListener:()=>{}};
  }}}});
  try {
    const update=scanWakeLock(); await update(true); await update(true); assert.equal(requests,1);
    document.hidden=true;events.visibilitychange();await new Promise(r=>setImmediate(r));assert.equal(releases,1);
    document.hidden=false;events.visibilitychange();await new Promise(r=>setImmediate(r));assert.equal(requests,2);
    await update(false);assert.equal(releases,2);
  } finally { for(const [k,d] of descriptors) {if(d)Object.defineProperty(globalThis,k,d);else Reflect.deleteProperty(globalThis,k);} }
});
