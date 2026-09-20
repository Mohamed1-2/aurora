"use strict";
/* AURORA world — the road you drive.
 *
 * Ported from the Drive-AI-3D reference (dependency-free WebGL) with one hard
 * rule change: the camera no longer runs on a simulated vehicle. Position is
 * owned by *real* telemetry — the odometer/speed the assistant is grounding on
 * (GET /v1/latest). Between polls the scene glides on the measured speed and
 * gently converges to the authoritative odometer, so the world moves exactly
 * as far as the vehicle moves. The cars/trees/guardrails are scenery only.
 */
window.AuroraWorld = (() => {
  const $ = id => document.getElementById(id);
  const canvas = $("roadCanvas");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const HUB = { posM: 214000, speed: 0, odoM: null, charging: false, paused: reduced.matches };

  const roadX = z => 17 * Math.sin(z / 220) + 31 * Math.sin(z / 520);
  const roadYaw = z => Math.atan(17 / 220 * Math.cos(z / 220) + 31 / 520 * Math.cos(z / 520));
  const hash = n => { const v = Math.sin(n * 127.1 + 311.7) * 43758.5453; return v - Math.floor(v); };
  let traffic = [];
  function seedTraffic() {
    traffic = [
      { z: 65, speed: 43, dir: 1, color: [.24, .35, .4] },
      { z: 230, speed: 57, dir: 1, color: [.76, .78, .73] },
      { z: 420, speed: 48, dir: 1, color: [.25, .29, .34] },
      { z: 130, speed: 57, dir: -1, color: [.75, .76, .78] },
      { z: 240, speed: 62, dir: -1, color: [.26, .39, .48] },
      { z: 370, speed: 48, dir: -1, color: [.49, .17, .15] },
      { z: 550, speed: 52, dir: -1, color: [.7, .7, .66] }
    ];
  }
  seedTraffic();

  const vertex = `
    attribute vec3 aPosition; attribute vec3 aNormal; attribute vec3 aColor;
    attribute vec2 aUV; attribute float aMaterial;
    uniform vec3 camera; uniform float yaw; uniform float aspect;
    varying vec3 color; varying vec3 normal; varying vec2 uv;
    varying float material; varying float distanceToCamera; varying vec3 world;
    void main(){
      vec3 p=aPosition-camera;
      float x=p.x*cos(yaw)-p.z*sin(yaw);
      float z=p.x*sin(yaw)+p.z*cos(yaw);
      gl_Position=vec4(x*1.75/aspect,p.y*1.75,z*1.00025-0.200025,z);
      color=aColor;normal=aNormal;uv=aUV;material=aMaterial;
      distanceToCamera=z;world=aPosition;
    }`;
  const fragment = `
    precision mediump float;
    varying vec3 color; varying vec3 normal; varying vec2 uv;
    varying float material; varying float distanceToCamera; varying vec3 world;
    uniform sampler2D tree;
    float noise(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
    void main(){
      vec3 c=color;
      if(material>1.5){vec4 tex=texture2D(tree,uv);if(tex.a<0.36)discard;c=tex.rgb*color;}
      else if(material>0.5){
        float grain=noise(uv*110.0);
        c*=0.88+grain*0.19;
        float edge=step(3.42,abs(uv.x))*(1.0-step(3.51,abs(uv.x)));
        float center=(1.0-step(.055,abs(uv.x)))*step(3.0,mod(uv.y,9.0));
        c=mix(c,vec3(.85,.86,.79),edge);
        c=mix(c,vec3(.88,.75,.30),center);
      }else{ c*=0.96+noise(world.xz*2.0)*.07; }
      if(material<1.5){float light=.65+.35*max(0.0,dot(normalize(normal),normalize(vec3(-.5,1.0,-.4))));c*=light;}
      float fog=smoothstep(95.0,650.0,distanceToCamera);
      c=mix(c,vec3(.67,.75,.76),fog);
      gl_FragColor=vec4(c,1.0);
    }`;
  const skyVertex = `attribute vec2 p;varying vec2 uv;void main(){uv=(p+1.0)*.5;gl_Position=vec4(p,.999,1.0);}`;
  const skyFragment = `
    precision mediump float; varying vec2 uv; uniform float t; uniform float direction;
    float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float n(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);}
    void main(){
      vec3 c=mix(vec3(.77,.83,.82),vec3(.35,.58,.73),smoothstep(.45,1.0,uv.y));
      vec2 p=vec2(uv.x*5.0+direction*.5+t*.0004,uv.y*4.0);
      float cloud=n(p)*.57+n(p*2.0)*.28+n(p*4.0)*.15;
      c=mix(c,vec3(.94,.94,.88),smoothstep(.56,.76,cloud)*smoothstep(.55,.8,uv.y)*.72);
      float sun=length((uv-vec2(.16-direction*.3,.82))*vec2(2.5,1.0));
      c+=vec3(.18,.12,.04)*exp(-sun*9.0);
      float mountain=.515+.025*sin(uv.x*21.0+direction)+.016*sin(uv.x*43.0+2.0)+.01*sin(uv.x*87.0);
      if(uv.y<mountain)c=mix(vec3(.42,.52,.53),vec3(.68,.76,.75),smoothstep(.47,.55,uv.y));
      gl_FragColor=vec4(c,1.0);
    }`;
  let gl, program, skyProgram, meshBuffer, skyBuffer, attrs, uniforms, skyUniforms;
  let treeReady = false, drawable = false, lost = false, frame = 0, previous = 0, accumulator = 0;
  let vertices = [];
  function shader(type, source) {
    const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function link(v, f) {
    const p = gl.createProgram(); gl.attachShader(p, shader(gl.VERTEX_SHADER, v)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, f)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  function setup() {
    try {
      gl = canvas.getContext('webgl', { alpha: false, antialias: true, powerPreference: 'high-performance' });
      if (!gl) throw new Error('WebGL unavailable');
      program = link(vertex, fragment); skyProgram = link(skyVertex, skyFragment);
      attrs = ['aPosition', 'aNormal', 'aColor', 'aUV', 'aMaterial'].map(x => gl.getAttribLocation(program, x));
      uniforms = Object.fromEntries(['camera', 'yaw', 'aspect'].map(x => [x, gl.getUniformLocation(program, x)]));
      skyUniforms = Object.fromEntries(['t', 'direction'].map(x => [x, gl.getUniformLocation(skyProgram, x)]));
      meshBuffer = gl.createBuffer(); skyBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, skyBuffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      const texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([60, 100, 40, 255]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const image = new Image();
      image.onload = () => { try { gl.bindTexture(gl.TEXTURE_2D, texture); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image); treeReady = true; } catch (e) { } };
      image.src = 'assets/tree.png';
      drawable = true; lost = false; resize(); canvas.style.opacity = '1';
    } catch (error) {
      drawable = false;
      document.body.classList.add('no3d');
    }
  }
  function add(p, n, c, uv = [0, 0], m = 0) { vertices.push(...p, ...n, ...c, ...uv, m); }
  function quad(a, b, c, d, color, normal = [0, 1, 0], material = 0, uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]) { for (const i of [0, 1, 2, 0, 2, 3]) add([a, b, c, d][i], normal, color, uvs[i], material); }
  function box(x, y, z, w, h, d, color, yaw = 0) {
    const co = Math.cos(yaw), si = Math.sin(yaw);
    const point = (px, py, pz) => [x + px * co + pz * si, y + py, z - px * si + pz * co];
    const p = [point(-w / 2, 0, -d / 2), point(w / 2, 0, -d / 2), point(w / 2, h, -d / 2), point(-w / 2, h, -d / 2), point(-w / 2, 0, d / 2), point(w / 2, 0, d / 2), point(w / 2, h, d / 2), point(-w / 2, h, d / 2)];
    for (const [idx, n] of [[[0, 1, 2, 3], [0, 0, -1]], [[5, 4, 7, 6], [0, 0, 1]], [[4, 0, 3, 7], [-1, 0, 0]], [[1, 5, 6, 2], [1, 0, 0]], [[3, 2, 6, 7], [0, 1, 0]]]) quad(...idx.map(i => p[i]), color, [n[0] * co + n[2] * si, n[1], -n[0] * si + n[2] * co]);
  }
  function car(c) {
    const z = c.z, x = roadX(z) + (c.dir === 1 ? -1.78 : 1.78), yaw = roadYaw(z) + (c.dir === -1 ? Math.PI : 0);
    const part = (ox, y, oz, w, h, d, col) => box(x + ox * Math.cos(yaw) + oz * Math.sin(yaw), y, z - ox * Math.sin(yaw) + oz * Math.cos(yaw), w, h, d, col, yaw);
    part(0, .27, 0, 1.78, .57, 4.2, c.color); part(0, .82, -.1, 1.45, .53, 1.98, c.color);
    part(0, .89, -1.11, 1.25, .35, .03, [.10, .19, .23]); part(0, .87, .91, 1.25, .36, .04, [.12, .22, .25]);
    part(-.737, .9, -.12, .02, .32, 1.62, [.10, .19, .22]); part(.737, .9, -.12, .02, .32, 1.62, [.10, .19, .22]);
    part(0, .31, -2.12, 1.63, .13, .07, [.13, .15, .15]); part(0, .45, -2.13, .37, .12, .03, [.78, .8, .73]);
    for (const side of [-1, 1]) {
      part(side * .67, .64, -2.11, .3, .14, .04, [.8, .06, .035]); part(side * .66, .64, 2.11, .35, .14, .04, [.92, .91, .71]);
      for (const axle of [-1.28, 1.25]) part(side * .88, .15, axle, .17, .56, .62, [.06, .07, .072]);
    }
  }
  function buildWorld() {
    vertices = [];
    const base = Math.floor(HUB.posM / 8) * 8 - 16;
    const bands = [-160, -35, -10, -4.0, -3.7, 3.7, 4.0, 10, 35, 160];
    for (let i = 0; i < 104; i++) {
      const z = base + i * 8, nz = z + 8;
      const terrain = (x, zz) => Math.abs(x) < 10 ? -.035 : Math.max(0, Math.sin(zz / 80 + x * .12) * 2.4 + (Math.abs(x) - 10) * .06);
      for (let j = 0; j < bands.length - 1; j++) {
        const l = bands[j], r = bands[j + 1], roadB = j === 4, shoulder = j === 3 || j === 5;
        const c = roadB ? [.235, .25, .26] : shoulder ? [.53, .52, .45] : [.30, .39, .22];
        const y = (v, zz) => roadB ? .012 : terrain(v, zz);
        quad([roadX(z) + l, y(l, z), z], [roadX(z) + r, y(r, z), z], [roadX(nz) + r, y(r, nz), nz], [roadX(nz) + l, y(l, nz), nz], c, [0, 1, 0], roadB ? 1 : 0, [[l, z], [r, z], [r, nz], [l, nz]]);
      }
    }
    for (let z = Math.floor(HUB.posM / 8) * 8; z < HUB.posM + 370; z += 8) {
      for (const side of [-1, 1]) {
        const x = roadX(z) + side * 4.65;
        box(x, .03, z, .11, .82, .12, [.46, .49, .48]);
        const end = roadX(z + 8) + side * 4.65;
        quad([x, .58, z], [x, .78, z], [end, .78, z + 8], [end, .58, z + 8], [.64, .65, .59], [side, 0, 0]);
      }
    }
    for (const c of traffic) if (c.z > HUB.posM - 8 && c.z < HUB.posM + 650) car(c);
    if (treeReady) {
      const start = Math.floor(HUB.posM / 13) - 2;
      for (let i = start + 48; i >= start; i--) {
        for (const side of [-1, 1]) {
          const seed = i * 7 + side * 71, z = i * 13 + hash(seed) * 8;
          const offset = 7 + hash(seed + 2) * 20, x = roadX(z) + side * offset;
          const height = 7 + hash(seed + 4) * 8, width = height;
          const yaw = roadYaw(HUB.posM), dx = Math.cos(yaw) * width / 2, dz = -Math.sin(yaw) * width / 2;
          const shade = .74 + hash(seed + 9) * .24;
          quad([x - dx, -.06, z - dz], [x + dx, -.06, z + dz], [x + dx, height, z + dz], [x - dx, height, z - dz], [shade, shade, shade], [0, 0, -1], 2, [[0, 1], [1, 1], [1, 0], [0, 0]]);
        }
      }
    }
  }
  function resize() {
    if (!gl || !drawable) return;
    const r = canvas.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 1.5, 1920 / Math.max(1, r.width));
    const w = Math.round(r.width * ratio), h = Math.round(r.height * ratio);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = Math.max(1, w); canvas.height = Math.max(1, h); }
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  function draw() {
    if (!drawable || lost) return;
    gl.clearColor(.67, .75, .76, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST); gl.useProgram(skyProgram); gl.bindBuffer(gl.ARRAY_BUFFER, skyBuffer);
    const p = gl.getAttribLocation(skyProgram, 'p');
    for (let i = 0; i < 8; i++) gl.disableVertexAttribArray(i);
    gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1f(skyUniforms.t, HUB.elapsedT || 0); gl.uniform1f(skyUniforms.direction, roadYaw(HUB.posM)); gl.drawArrays(gl.TRIANGLES, 0, 6);
    buildWorld(); gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.useProgram(program); gl.bindBuffer(gl.ARRAY_BUFFER, meshBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
    for (let i = 0; i < 8; i++) gl.disableVertexAttribArray(i);
    const sizes = [3, 3, 3, 2, 1], offsets = [0, 3, 6, 9, 11];
    attrs.forEach((a, i) => { gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, sizes[i], gl.FLOAT, false, 48, offsets[i] * 4); });
    const bob = HUB.speed > 0.5 ? Math.sin((HUB.elapsedT || 0) * 10) * .009 * (Math.min(HUB.speed, 120) / 60) : 0;
    gl.uniform3f(uniforms.camera, roadX(HUB.posM) - 1.4, 1.38 + bob, HUB.posM);
    gl.uniform1f(uniforms.yaw, roadYaw(HUB.posM + 12)); gl.uniform1f(uniforms.aspect, canvas.width / canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 12);
  }
  function tick(dt) {
    if (HUB.odoM !== null && HUB.odoM !== HUB.lastOdoM) {
      HUB.lastOdoM = HUB.odoM;
    }
    HUB.elapsedT = (HUB.elapsedT || 0) + dt;
    for (const c of traffic) {
      c.z += c.dir * c.speed / 3.6 * dt;
      if (c.z < HUB.posM - 25) c.z = HUB.posM + 520 + hash(c.speed + HUB.posM) * 140;
      if (c.z > HUB.posM + 900) c.z = HUB.posM + 500;
    }
    if (HUB.charging) { HUB.speed = 0; return; }
    HUB.posM += HUB.speed / 3.6 * dt;
    if (HUB.odoM !== null) HUB.posM += (HUB.odoM - HUB.posM) * Math.min(1, dt * 2.2);
  }
  function render(now) {
    const dt = previous ? Math.min((now - previous) / 1000, .05) : 0;
    previous = now;
    if (!document.hidden && !HUB.paused) { tick(dt); draw(); }
    frame = requestAnimationFrame(render);
  }
  window.addEventListener('resize', resize);
  canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); lost = true; canvas.style.opacity = '0'; });
  canvas.addEventListener('webglcontextrestored', () => { setup(); });
  window.addEventListener('pagehide', () => cancelAnimationFrame(frame));
  window.addEventListener('pageshow', e => { if (e.persisted) { previous = 0; frame = requestAnimationFrame(render); } });
  reduced.addEventListener('change', e => { HUB.paused = e.matches; });

  return {
    start() {
      setup();
      frame = requestAnimationFrame(render);
    },
    sync(speedKph, odoKm, charging) {
      HUB.speed = Number.isFinite(speedKph) && speedKph > 0 ? speedKph : 0;
      HUB.odoM = odoKm !== null && Number.isFinite(odoKm) ? odoKm * 1000 : null;
      HUB.charging = !!charging;
    },
    pause() { HUB.paused = true; },
    resume() { if (!reduced.matches) HUB.paused = false; },
    get ready() { return drawable; }
  };
})();