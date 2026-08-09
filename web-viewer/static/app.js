import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";


const elements = {
  viewport: document.querySelector("#viewport"),
  canvas: document.querySelector("#viewer-canvas"),
  boneTooltip: document.querySelector("#bone-tooltip"),
  loading: document.querySelector("#loading-overlay"),
  loadingText: document.querySelector("#loading-text"),
  status: document.querySelector("#status"),
  statusText: document.querySelector("#status-text"),
  activeParameter: document.querySelector("#active-parameter"),
  vertexCount: document.querySelector("#vertex-count"),
  faceCount: document.querySelector("#face-count"),
  jointCount: document.querySelector("#joint-count"),
  renderTime: document.querySelector("#render-time"),
  poseCount: document.querySelector("#pose-count"),
  identityCount: document.querySelector("#identity-count"),
  expressionCount: document.querySelector("#expression-count"),
  groupTabs: [...document.querySelectorAll(".tab[data-group]")],
  search: document.querySelector("#parameter-search"),
  categoryField: document.querySelector("#category-field"),
  category: document.querySelector("#category-select"),
  parameter: document.querySelector("#parameter-select"),
  filteredCount: document.querySelector("#filtered-count"),
  slider: document.querySelector("#parameter-slider"),
  value: document.querySelector("#value-readout"),
  rangeReadout: document.querySelector("#range-readout"),
  jointsField: document.querySelector("#joints-field"),
  jointList: document.querySelector("#joint-list"),
  resetCurrent: document.querySelector("#reset-current"),
  resetGroup: document.querySelector("#reset-group"),
  resetAll: document.querySelector("#reset-all"),
  bodyColor: document.querySelector("#body-color"),
  backgroundColor: document.querySelector("#background-color"),
  wireframe: document.querySelector("#wireframe-toggle"),
  skeleton: document.querySelector("#skeleton-toggle"),
  grid: document.querySelector("#grid-toggle"),
  axes: document.querySelector("#axes-toggle"),
  cameraReset: document.querySelector("#camera-reset"),
  projection: document.querySelector("#projection-toggle"),
  viewButtons: [...document.querySelectorAll("[data-view]")],
  screenshot: document.querySelector("#screenshot-button"),
  fullscreen: document.querySelector("#fullscreen-button"),
  correctives: document.querySelector("#correctives-toggle"),
  correctivesState: document.querySelector("#correctives-state"),
  lod: document.querySelector("#lod-select"),
  lodSummary: document.querySelector("#lod-summary"),
  exportFormat: document.querySelector("#export-format"),
  exportGround: document.querySelector("#export-ground-toggle"),
  exportGroundState: document.querySelector("#export-ground-state"),
  exportButton: document.querySelector("#export-button"),
};


const state = {
  metadata: null,
  currentGroup: "pose",
  selectedIndex: {
    pose: 0,
    identity: 0,
    expression: 0,
  },
  requestPending: false,
  bounds: null,
  jointPositions: null,
  projectedJoints: null,
  projection: "perspective",
  lastView: "front",
};

const VIEW_DIRECTIONS = {
  front: new THREE.Vector3(0, 0, 1),
  back: new THREE.Vector3(0, 0, -1),
  left: new THREE.Vector3(-1, 0, 0),
  right: new THREE.Vector3(1, 0, 0),
};
const JOINT_HOVER_RADIUS_PX = 10;
const JOINT_OVERLAP_TOLERANCE_PX = 0.5;


const renderer = new THREE.WebGLRenderer({
  canvas: elements.canvas,
  antialias: true,
  alpha: false,
  preserveDrawingBuffer: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.setClearColor(elements.backgroundColor.value, 1);

const scene = new THREE.Scene();
const perspectiveCamera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
const orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
let activeCamera = perspectiveCamera;
let controls = new OrbitControls(activeCamera, elements.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = true;

const hemisphereLight = new THREE.HemisphereLight(0xdde8ff, 0x283040, 2.15);
scene.add(hemisphereLight);
const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
keyLight.position.set(3, 4, 5);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x7798ff, 1.4);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

const bodyMaterial = new THREE.MeshStandardMaterial({
  color: elements.bodyColor.value,
  roughness: 0.58,
  metalness: 0.02,
  side: THREE.DoubleSide,
});
const bodyMesh = new THREE.Mesh(new THREE.BufferGeometry(), bodyMaterial);
scene.add(bodyMesh);

const skeletonMaterial = new THREE.LineBasicMaterial({
  color: 0xffc45c,
  transparent: true,
  opacity: 0.95,
  depthTest: false,
});
const skeletonLines = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  skeletonMaterial,
);
skeletonLines.renderOrder = 3;
const skeletonPoints = new THREE.Points(
  new THREE.BufferGeometry(),
  new THREE.PointsMaterial({
    color: 0xfff1bd,
    size: 0.045,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.95,
    map: createDotTexture(),
    alphaTest: 0.5,
    depthTest: false,
  }),
);
skeletonPoints.renderOrder = 4;
const skeletonOverlay = new THREE.Group();
skeletonOverlay.add(skeletonLines, skeletonPoints);
skeletonOverlay.visible = false;
scene.add(skeletonOverlay);
const skeletonProjection = new THREE.Vector3();

const gridHelper = new THREE.GridHelper(4, 20, 0x526077, 0x2b3442);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.52;
scene.add(gridHelper);

const axesHelper = new THREE.AxesHelper(0.45);
axesHelper.visible = false;
scene.add(axesHelper);


function formatNumber(value, digits = 4) {
  return Number(value).toFixed(digits);
}


function formatCount(value) {
  return new Intl.NumberFormat().format(value);
}


function createDotTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext("2d");
  context.fillStyle = "#ffffff";
  context.beginPath();
  context.arc(16, 16, 14, 0, Math.PI * 2);
  context.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}


async function fetchChecked(resource, options, fallbackMessage) {
  const response = await fetch(resource, options);
  if (response.ok) return response;

  let message = fallbackMessage;
  try {
    const payload = await response.json();
    message = payload.error || message;
  } catch {
    // The backend may return an HTML or empty error response.
  }
  throw new Error(message);
}


function updateInferenceTime(response) {
  const inferenceMs = Number(response.headers.get("X-MHR-Inference-Ms"));
  elements.renderTime.textContent = Number.isFinite(inferenceMs)
    ? `${formatNumber(inferenceMs, 1)} ms`
    : "—";
}


function setStatus(message, kind = "ready") {
  elements.statusText.textContent = message;
  elements.status.classList.toggle("is-busy", kind === "busy");
  elements.status.classList.toggle("is-error", kind === "error");
}


function showLoading(message) {
  elements.loadingText.textContent = message;
  elements.loading.classList.remove("is-hidden");
}


function hideLoading() {
  elements.loading.classList.add("is-hidden");
}


function setInputsDisabled(disabled) {
  state.requestPending = disabled;
  [
    elements.resetCurrent,
    elements.resetGroup,
    elements.resetAll,
    elements.correctives,
    elements.lod,
    elements.exportFormat,
    elements.exportGround,
    elements.exportButton,
  ].forEach((element) => {
    element.disabled = disabled;
  });
  elements.parameter.setAttribute("aria-disabled", String(disabled));
  elements.parameter.querySelectorAll(".parameter-item").forEach((item) => {
    item.disabled = disabled;
  });
  const parameter = state.metadata ? currentParameter() : null;
  const locked = parameter ? isLocked(parameter) : false;
  elements.slider.disabled = disabled || locked;
  elements.value.disabled = disabled || locked;
  elements.resetCurrent.disabled = disabled || locked;
}


function updateModelStatistics() {
  const model = state.metadata.model;
  elements.vertexCount.textContent = formatCount(model.vertexCount);
  elements.faceCount.textContent = formatCount(model.faceCount);
  elements.jointCount.textContent = formatCount(model.jointCount);
}


function updateModelControls() {
  const model = state.metadata.model;
  if (elements.lod.options.length !== model.supportedLods.length) {
    elements.lod.replaceChildren();
    for (const entry of model.supportedLods) {
      const option = document.createElement("option");
      option.value = String(entry.lod);
      option.textContent = `LOD ${entry.lod}`;
      elements.lod.appendChild(option);
    }
  }
  elements.lod.value = String(model.lod);
  elements.correctives.classList.toggle("is-on", model.applyCorrectives);
  elements.correctives.setAttribute(
    "aria-checked",
    String(model.applyCorrectives),
  );
  elements.correctivesState.textContent = model.applyCorrectives ? "On" : "Off";
  elements.exportGround.classList.toggle("is-on", model.snapToGround);
  elements.exportGround.setAttribute(
    "aria-checked",
    String(model.snapToGround),
  );
  elements.exportGroundState.textContent = model.snapToGround ? "On" : "Off";
  elements.lodSummary.textContent =
    `LOD ${model.lod} · ${formatCount(model.vertexCount)} vertices · ` +
    `${formatCount(model.faceCount)} faces`;

  if (elements.exportFormat.options.length !== model.exportFormats.length) {
    elements.exportFormat.replaceChildren();
    for (const fileFormat of model.exportFormats) {
      const option = document.createElement("option");
      option.value = fileFormat;
      option.textContent = `.${fileFormat.toUpperCase()}`;
      elements.exportFormat.appendChild(option);
    }
  }
}


function currentParameters() {
  return state.metadata.groups[state.currentGroup];
}


function currentParameter() {
  return currentParameters()[state.selectedIndex[state.currentGroup]];
}


function isLocked(parameter) {
  return parameter.min === parameter.max || parameter.managed === true;
}


function buildCategories() {
  const categories = [
    "All categories",
    ...new Set(state.metadata.groups.pose.map((parameter) => parameter.category)),
  ];
  elements.category.replaceChildren();
  for (const category of categories) {
    const option = document.createElement("option");
    option.value = category === "All categories" ? "" : category;
    option.textContent = category;
    elements.category.appendChild(option);
  }
}


function filteredParameters() {
  const query = elements.search.value.trim().toLowerCase();
  const category = elements.category.value;
  return currentParameters().filter((parameter) => {
    const matchesQuery =
      !query ||
      parameter.name.toLowerCase().includes(query) ||
      parameter.label.toLowerCase().includes(query);
    const matchesCategory =
      state.currentGroup !== "pose" || !category || parameter.category === category;
    return matchesQuery && matchesCategory;
  });
}


function rebuildParameterList() {
  const selected = currentParameter();
  const filtered = filteredParameters();
  elements.parameter.replaceChildren();

  for (const parameter of filtered) {
    const locked = isLocked(parameter);
    const item = document.createElement("button");
    item.type = "button";
    item.className = `parameter-item${locked ? " is-locked" : ""}`;
    item.dataset.index = String(parameter.index);
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", "false");
    if (locked) {
      const reason = parameter.lockedReason || "Locked by model";
      item.title = reason;
      item.setAttribute("aria-label", `${parameter.label}, ${reason}`);
    }

    const label = document.createElement("span");
    label.className = "parameter-item-label";
    label.textContent = state.currentGroup === "pose"
      ? parameter.name
      : parameter.label;
    item.appendChild(label);

    if (state.currentGroup === "pose") {
      const lockSlot = document.createElement("span");
      lockSlot.className = "parameter-lock-slot";
      if (locked) {
        const lock = document.createElement("span");
        lock.className = "parameter-lock";
        lock.setAttribute("aria-hidden", "true");
        lockSlot.appendChild(lock);
      }
      item.appendChild(lockSlot);

      const category = document.createElement("span");
      category.className = "parameter-item-category";
      category.textContent = parameter.category;
      item.appendChild(category);
    } else if (locked) {
      const lock = document.createElement("span");
      lock.className = "parameter-lock";
      lock.setAttribute("aria-hidden", "true");
      item.appendChild(lock);
    }
    elements.parameter.appendChild(item);
  }

  elements.filteredCount.textContent = `${filtered.length} shown`;
  let next = selected && filtered.some((item) => item.index === selected.index)
    ? selected
    : filtered[0];

  if (!next) {
    elements.slider.disabled = true;
    elements.value.disabled = true;
    elements.parameter.setAttribute("aria-disabled", "true");
    elements.value.value = "";
    elements.rangeReadout.textContent = "—";
    elements.jointList.textContent = "No matching parameters";
    return;
  }

  state.selectedIndex[state.currentGroup] = next.index;
  elements.parameter.setAttribute("aria-disabled", String(state.requestPending));
  elements.parameter.querySelectorAll(".parameter-item").forEach((item) => {
    const selectedItem = Number(item.dataset.index) === next.index;
    item.classList.toggle("is-selected", selectedItem);
    item.setAttribute("aria-selected", String(selectedItem));
    item.disabled = state.requestPending;
    if (selectedItem) item.scrollIntoView({ block: "nearest" });
  });
  showParameter(next);
}


function showParameter(parameter) {
  const locked = isLocked(parameter);
  const span = Math.max(parameter.max - parameter.min, 0.0001);
  const step = Math.max(span / 600, 0.0001);
  elements.slider.min = parameter.min;
  elements.slider.max = parameter.max;
  elements.slider.step = step;
  elements.slider.value = parameter.value;
  elements.value.min = parameter.min;
  elements.value.max = parameter.max;
  elements.value.step = step;
  elements.value.value = formatNumber(parameter.value);
  elements.rangeReadout.textContent =
    `${formatNumber(parameter.min, 2)} … ${formatNumber(parameter.max, 2)}`;
  elements.activeParameter.textContent =
    `${state.currentGroup} / ${parameter.name} = ${formatNumber(parameter.value)}`;
  elements.jointsField.hidden = state.currentGroup !== "pose";
  elements.jointList.textContent = parameter.joints.length
    ? parameter.joints.join(", ")
    : "No directly affected joints";
  elements.slider.disabled = state.requestPending || locked;
  elements.value.disabled = state.requestPending || locked;
  elements.resetCurrent.disabled = state.requestPending || locked;
}


function selectGroup(group) {
  state.currentGroup = group;
  for (const tab of elements.groupTabs) {
    const active = tab.dataset.group === group;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-current", active ? "true" : "false");
  }
  elements.categoryField.hidden = group !== "pose";
  elements.search.value = "";
  rebuildParameterList();
}


function parseGeometry(buffer) {
  const values = new Float32Array(buffer);
  const vertexFloatCount = state.metadata.model.vertexCount * 3;
  const jointFloatCount = state.metadata.model.jointCount * 3;
  if (values.length !== vertexFloatCount + jointFloatCount) {
    throw new Error(
      `Unexpected geometry payload: ${values.length} floats received`,
    );
  }
  return {
    vertices: values.slice(0, vertexFloatCount),
    joints: values.slice(vertexFloatCount, vertexFloatCount + jointFloatCount),
  };
}


function updateSkeleton(joints) {
  const lineValues = [];
  state.jointPositions = joints;
  state.projectedJoints = new Float32Array(joints.length);
  state.metadata.jointParents.forEach((parent, child) => {
    if (parent < 0 || parent === child) return;
    lineValues.push(
      joints[child * 3],
      joints[child * 3 + 1],
      joints[child * 3 + 2],
      joints[parent * 3],
      joints[parent * 3 + 1],
      joints[parent * 3 + 2],
    );
  });
  skeletonLines.geometry.dispose();
  skeletonLines.geometry = new THREE.BufferGeometry();
  skeletonLines.geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(lineValues, 3),
  );
  skeletonPoints.geometry.dispose();
  skeletonPoints.geometry = new THREE.BufferGeometry();
  skeletonPoints.geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(joints, 3),
  );
}


function hideBoneTooltip() {
  elements.boneTooltip.hidden = true;
}


function showBoneTooltip(event, names) {
  elements.boneTooltip.replaceChildren(
    ...names.map((name) => {
      const row = document.createElement("div");
      row.textContent = name;
      return row;
    }),
  );
  elements.boneTooltip.hidden = false;
  const viewportRect = elements.viewport.getBoundingClientRect();
  const margin = 8;
  const offset = 14;
  const left = Math.min(
    event.clientX - viewportRect.left + offset,
    elements.viewport.clientWidth - elements.boneTooltip.offsetWidth - margin,
  );
  const top = Math.min(
    event.clientY - viewportRect.top + offset,
    elements.viewport.clientHeight - elements.boneTooltip.offsetHeight - margin,
  );
  elements.boneTooltip.style.left = `${Math.max(margin, left)}px`;
  elements.boneTooltip.style.top = `${Math.max(margin, top)}px`;
}


function updateBoneTooltip(event) {
  if (
    !elements.skeleton.checked
    || event.buttons !== 0
    || !state.jointPositions
  ) {
    hideBoneTooltip();
    return;
  }

  const canvasRect = elements.canvas.getBoundingClientRect();
  const pointerX = event.clientX - canvasRect.left;
  const pointerY = event.clientY - canvasRect.top;
  const projected = state.projectedJoints;
  for (let joint = 0; joint < state.metadata.model.jointCount; joint += 1) {
    skeletonProjection
      .fromArray(state.jointPositions, joint * 3)
      .project(activeCamera);
    projected[joint * 3] =
      (skeletonProjection.x * 0.5 + 0.5) * canvasRect.width;
    projected[joint * 3 + 1] =
      (-skeletonProjection.y * 0.5 + 0.5) * canvasRect.height;
    projected[joint * 3 + 2] = skeletonProjection.z;
  }

  let closestJoint = -1;
  let closestDistance = JOINT_HOVER_RADIUS_PX ** 2;
  for (let joint = 0; joint < state.metadata.model.jointCount; joint += 1) {
    const offset = joint * 3;
    if (Math.abs(projected[offset + 2]) > 1) continue;
    const distance =
      (pointerX - projected[offset]) ** 2
      + (pointerY - projected[offset + 1]) ** 2;
    if (distance < closestDistance) {
      closestDistance = distance;
      closestJoint = joint;
    }
  }

  if (closestJoint < 0) {
    hideBoneTooltip();
    return;
  }

  const closestOffset = closestJoint * 3;
  const overlapDistance = JOINT_OVERLAP_TOLERANCE_PX ** 2;
  const names = [];
  for (let joint = 0; joint < state.metadata.model.jointCount; joint += 1) {
    const offset = joint * 3;
    if (Math.abs(projected[offset + 2]) > 1) continue;
    const distance =
      (projected[offset] - projected[closestOffset]) ** 2
      + (projected[offset + 1] - projected[closestOffset + 1]) ** 2;
    if (distance <= overlapDistance) {
      names.push(state.metadata.jointNames[joint]);
    }
  }
  showBoneTooltip(event, names);
}


function replaceTopology(topology) {
  const previousGeometry = bodyMesh.geometry;
  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(new THREE.BufferAttribute(topology, 1));
  bodyMesh.geometry = geometry;
  previousGeometry.dispose();
}


function updateGeometry(payload, { fit = false } = {}) {
  hideBoneTooltip();
  const position = bodyMesh.geometry.getAttribute("position");
  if (!position || position.array.length !== payload.vertices.length) {
    bodyMesh.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(payload.vertices, 3),
    );
    bodyMesh.geometry.deleteAttribute("normal");
  } else {
    position.array.set(payload.vertices);
    position.needsUpdate = true;
  }
  bodyMesh.geometry.computeVertexNormals();
  bodyMesh.geometry.computeBoundingBox();
  bodyMesh.geometry.computeBoundingSphere();
  state.bounds = bodyMesh.geometry.boundingBox.clone();
  updateSkeleton(payload.joints);
  if (fit) fitCamera("front");
}


function cameraMetrics() {
  if (!state.bounds) return null;
  const size = state.bounds.getSize(new THREE.Vector3());
  const center = state.bounds.getCenter(new THREE.Vector3());
  const maxDimension = Math.max(size.x, size.y, size.z);
  return { size, center, maxDimension };
}


function viewDirection(view) {
  return VIEW_DIRECTIONS[view] ?? VIEW_DIRECTIONS.front;
}


function fitCamera(view = state.lastView) {
  const metrics = cameraMetrics();
  if (!metrics) return;
  state.lastView = view;
  const direction = viewDirection(view);
  const distance =
    (metrics.maxDimension / (2 * Math.tan(THREE.MathUtils.degToRad(32 / 2)))) * 1.18;

  perspectiveCamera.position
    .copy(metrics.center)
    .addScaledVector(direction, distance);
  perspectiveCamera.near = Math.max(distance / 1000, 0.001);
  perspectiveCamera.far = distance * 20;
  perspectiveCamera.updateProjectionMatrix();

  const half = metrics.maxDimension * 0.62;
  orthographicCamera.left = -half;
  orthographicCamera.right = half;
  orthographicCamera.top = half;
  orthographicCamera.bottom = -half;
  orthographicCamera.position
    .copy(metrics.center)
    .addScaledVector(direction, distance);
  orthographicCamera.near = Math.max(distance / 1000, 0.001);
  orthographicCamera.far = distance * 20;
  orthographicCamera.updateProjectionMatrix();

  controls.target.copy(metrics.center);
  activeCamera.lookAt(metrics.center);
  controls.update();
}


function replaceControls(camera) {
  const target = controls.target.clone();
  controls.dispose();
  activeCamera = camera;
  controls = new OrbitControls(activeCamera, elements.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.screenSpacePanning = true;
  controls.target.copy(target);
  controls.update();
}


function toggleProjection() {
  if (state.projection === "perspective") {
    state.projection = "orthographic";
    replaceControls(orthographicCamera);
    elements.projection.textContent = "Perspective";
  } else {
    state.projection = "perspective";
    replaceControls(perspectiveCamera);
    elements.projection.textContent = "Orthographic";
  }
  fitCamera(state.lastView);
}


async function deform(action, value = null) {
  if (state.requestPending) return;
  const parameter = currentParameter();
  setInputsDisabled(true);
  setStatus("Updating geometry…", "busy");

  try {
    const response = await fetchChecked(
      "/api/deform",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          group: state.currentGroup,
          index: parameter.index,
          value,
        }),
      },
      "The model update failed",
    );

    const geometry = parseGeometry(await response.arrayBuffer());
    updateGeometry(geometry);
    updateInferenceTime(response);

    if (action === "set") {
      parameter.value = Math.min(
        parameter.max,
        Math.max(parameter.min, Number(value)),
      );
    } else if (action === "reset_current") {
      parameter.value = Math.min(parameter.max, Math.max(parameter.min, 0));
    } else if (action === "reset_group") {
      currentParameters().forEach((item) => {
        item.value = Math.min(item.max, Math.max(item.min, 0));
      });
    } else if (action === "reset_all") {
      Object.values(state.metadata.groups).flat().forEach((item) => {
        item.value = Math.min(item.max, Math.max(item.min, 0));
      });
    }
    const rootTy = Number(response.headers.get("X-MHR-Root-TY"));
    const rootParameter = state.metadata.groups.pose.find(
      (item) => item.name === "root_ty",
    );
    if (rootParameter && Number.isFinite(rootTy)) {
      rootParameter.value = rootTy;
    }
    showParameter(parameter);
    setStatus(`Updated ${parameter.name}`);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setInputsDisabled(false);
  }
}


function commitParameterValue() {
  const value = elements.value.valueAsNumber;
  if (Number.isFinite(value)) {
    deform("set", value);
  } else {
    showParameter(currentParameter());
  }
}


async function configureModel(changes) {
  if (state.requestPending) return;
  const previousLod = state.metadata.model.lod;
  const requestedLod = changes.lod ?? previousLod;
  const lodChanged = requestedLod !== previousLod;
  const snapChanged =
    changes.snapToGround !== undefined &&
    changes.snapToGround !== state.metadata.model.snapToGround;
  setInputsDisabled(true);
  setStatus(
    lodChanged
      ? `Switching to LOD ${requestedLod}…`
      : snapChanged
        ? "Updating ground snap…"
        : "Updating pose correctives…",
    "busy",
  );
  if (lodChanged) showLoading(`Building LOD ${requestedLod} mesh…`);

  try {
    const response = await fetchChecked(
      "/api/configure",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      },
      "The model configuration failed",
    );

    const metadata = await response.json();
    let topology = null;
    if (lodChanged) {
      const topologyResponse = await fetchChecked(
        `/api/topology?lod=${metadata.model.lod}&revision=${metadata.revision}`,
        { cache: "no-store" },
        "The viewer backend did not return the LOD topology.",
      );
      topology = new Uint32Array(await topologyResponse.arrayBuffer());
    }

    const geometryResponse = await fetchChecked(
      `/api/geometry?revision=${metadata.revision}`,
      { cache: "no-store" },
      "The viewer backend did not return the configured geometry.",
    );

    state.metadata = metadata;
    if (topology) {
      replaceTopology(topology);
    }
    const geometry = parseGeometry(await geometryResponse.arrayBuffer());
    updateGeometry(geometry, { fit: lodChanged });
    updateInferenceTime(geometryResponse);
    updateModelStatistics();
    updateModelControls();
    rebuildParameterList();
    setStatus(
      lodChanged
        ? `LOD ${metadata.model.lod} ready`
        : snapChanged
          ? `Snap to ground ${metadata.model.snapToGround ? "enabled" : "disabled"}`
          : `Pose correctives ${metadata.model.applyCorrectives ? "enabled" : "disabled"}`,
    );
  } catch (error) {
    updateModelControls();
    setStatus(error.message, "error");
  } finally {
    if (lodChanged) hideLoading();
    setInputsDisabled(false);
  }
}


function downloadFilename(response, fallback) {
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^";]+)"?/i);
  return match ? match[1] : fallback;
}


async function exportCurrentModel() {
  if (state.requestPending) return;
  const fileFormat = elements.exportFormat.value;
  setInputsDisabled(true);
  setStatus(`Preparing ${fileFormat.toUpperCase()} export…`, "busy");

  try {
    const response = await fetchChecked(
      `/api/export/${fileFormat}`,
      { cache: "no-store" },
      "The model export failed",
    );

    const blob = await response.blob();
    const filename = downloadFilename(
      response,
      `mhr-lod${state.metadata.model.lod}.${fileFormat}`,
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setStatus(`Downloaded ${filename}`);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setInputsDisabled(false);
  }
}


function resizeRenderer() {
  const width = Math.max(1, Math.round(elements.viewport.clientWidth));
  const height = Math.max(1, Math.round(elements.viewport.clientHeight));
  renderer.setSize(width, height, false);
  const aspect = width / height;
  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  const verticalHalf = (orthographicCamera.top - orthographicCamera.bottom) / 2;
  orthographicCamera.left = -verticalHalf * aspect;
  orthographicCamera.right = verticalHalf * aspect;
  orthographicCamera.updateProjectionMatrix();
}


async function initialize() {
  try {
    const metadataResponse = await fetchChecked(
      "/api/metadata",
      { cache: "no-store" },
      "The viewer backend did not return model metadata.",
    );
    state.metadata = await metadataResponse.json();
    const revision = state.metadata.revision;
    const [topologyResponse, geometryResponse] = await Promise.all([
      fetchChecked(
        `/api/topology?revision=${revision}`,
        { cache: "no-store" },
        "The viewer backend did not return model topology.",
      ),
      fetchChecked(
        `/api/geometry?revision=${revision}`,
        { cache: "no-store" },
        "The viewer backend did not return model geometry.",
      ),
    ]);

    const topology = new Uint32Array(await topologyResponse.arrayBuffer());
    const geometry = parseGeometry(await geometryResponse.arrayBuffer());

    replaceTopology(topology);
    updateGeometry(geometry, { fit: true });

    updateModelStatistics();
    updateModelControls();
    elements.poseCount.textContent = state.metadata.groups.pose.length;
    elements.identityCount.textContent = state.metadata.groups.identity.length;
    elements.expressionCount.textContent = state.metadata.groups.expression.length;
    updateInferenceTime(geometryResponse);

    buildCategories();
    const preferred = state.metadata.groups.pose.find(
      (parameter) => parameter.name === "head_lean",
    );
    if (preferred) state.selectedIndex.pose = preferred.index;
    selectGroup("pose");
    hideLoading();
    setStatus("Ready");
  } catch (error) {
    elements.loadingText.textContent = error.message;
    setStatus(error.message, "error");
  }
}


elements.groupTabs.forEach((tab) => {
  tab.addEventListener("click", () => selectGroup(tab.dataset.group));
});
elements.search.addEventListener("input", () => rebuildParameterList());
elements.category.addEventListener("change", () => rebuildParameterList());
elements.parameter.addEventListener("click", (event) => {
  const item = event.target.closest(".parameter-item");
  if (!item || state.requestPending) return;
  state.selectedIndex[state.currentGroup] = Number(item.dataset.index);
  elements.parameter.querySelectorAll(".parameter-item").forEach((candidate) => {
    const selected = candidate === item;
    candidate.classList.toggle("is-selected", selected);
    candidate.setAttribute("aria-selected", String(selected));
  });
  showParameter(currentParameter());
});
elements.slider.addEventListener("input", () => {
  elements.value.value = formatNumber(elements.slider.value);
});
elements.slider.addEventListener("change", () => {
  deform("set", Number(elements.slider.value));
});
elements.value.addEventListener("change", commitParameterValue);
elements.value.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  commitParameterValue();
});
elements.resetCurrent.addEventListener("click", () => deform("reset_current"));
elements.resetGroup.addEventListener("click", () => deform("reset_group"));
elements.resetAll.addEventListener("click", () => deform("reset_all"));
elements.correctives.addEventListener("click", () => {
  configureModel({
    applyCorrectives: !state.metadata.model.applyCorrectives,
  });
});
elements.lod.addEventListener("change", () => {
  configureModel({ lod: Number(elements.lod.value) });
});
elements.exportGround.addEventListener("click", () => {
  configureModel({
    snapToGround: !state.metadata.model.snapToGround,
  });
});
elements.exportButton.addEventListener("click", exportCurrentModel);

elements.bodyColor.addEventListener("input", () => {
  bodyMaterial.color.set(elements.bodyColor.value);
});
elements.backgroundColor.addEventListener("input", () => {
  renderer.setClearColor(elements.backgroundColor.value, 1);
  elements.viewport.style.background = elements.backgroundColor.value;
});
elements.wireframe.addEventListener("change", () => {
  bodyMaterial.wireframe = elements.wireframe.checked;
});
elements.skeleton.addEventListener("change", () => {
  skeletonOverlay.visible = elements.skeleton.checked;
  if (!elements.skeleton.checked) hideBoneTooltip();
});
elements.grid.addEventListener("change", () => {
  gridHelper.visible = elements.grid.checked;
});
elements.axes.addEventListener("change", () => {
  axesHelper.visible = elements.axes.checked;
});
elements.canvas.addEventListener("pointermove", updateBoneTooltip);
elements.canvas.addEventListener("pointerdown", hideBoneTooltip);
elements.canvas.addEventListener("pointerleave", hideBoneTooltip);
document.addEventListener("pointermove", (event) => {
  if (event.target !== elements.canvas) hideBoneTooltip();
});
window.addEventListener("blur", hideBoneTooltip);

elements.viewButtons.forEach((button) => {
  button.addEventListener("click", () => fitCamera(button.dataset.view));
});
elements.cameraReset.addEventListener("click", () => fitCamera(state.lastView));
elements.projection.addEventListener("click", toggleProjection);

elements.screenshot.addEventListener("click", () => {
  renderer.render(scene, activeCamera);
  const link = document.createElement("a");
  link.download = `mhr-${state.currentGroup}-${currentParameter().name}.png`;
  link.href = elements.canvas.toDataURL("image/png");
  link.click();
});
elements.fullscreen.addEventListener("click", async () => {
  if (!document.fullscreenElement) {
    await elements.viewport.requestFullscreen();
  } else {
    await document.exitFullscreen();
  }
  resizeRenderer();
});

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, button")) return;
  if (event.key.toLowerCase() === "f") fitCamera("front");
  if (event.key.toLowerCase() === "r") fitCamera(state.lastView);
});

new ResizeObserver(resizeRenderer).observe(elements.viewport);
resizeRenderer();

function animate() {
  controls.update();
  renderer.render(scene, activeCamera);
  requestAnimationFrame(animate);
}
animate();
initialize();
