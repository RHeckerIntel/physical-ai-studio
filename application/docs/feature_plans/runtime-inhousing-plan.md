# Runtime Inhousing Plan

## Why?

The current physicalai runtime does not allow for the same flexibility that is needed in the studio. The studio needs to be able to switch models, datasets, run human-in-the-loop. Reimplementing a flexible version of the Runtime session into physicalai would be possible, but it might be too complicated for the use case in physicalai, where the main goal is to run a model on the robot.

Specifically the planned "Control" feature will be extremely handy for many of the issues that we are facing when implementing new stuff in the robot.

## Plan

Since this will be a rather big refactor we will work on it in steps using a feature branch, only merging said feature branch at the end.
The current implementation of runtime is first removed and the endpoints made into placeholders, or buttons disabled. After that reimplementation is gradually added.
This is done because a gradual change is very hard and the feature-branch approach allows for incomplete or broken states of the feature branch.
A detailed AI implementation plan exists in the second part of this document.

Now onto the architecture. A new RuntimeSession is created. This session is responsible for handling lifecycle of the session. This session can load an environment, dataset, model and control.

## Lifecycle handling

A lot of components in these acquire real hardware. This means that it needs to be acquired properly and released once done.
Same goes for Processes that are started which need to be terminated when the RuntimeSession stops.

In order to do this we will use contextmanagers. The goal is to have the lifecycle be handled by the thread or process itself:

```python

class EnvironmentContext:
    follower: RobotWorker
    leader: RobotWorker | None
    cameras: dict[str, CameraWorker]

class EnvironmentWorker(BaseThreadWorker[EnvironmentContext]):
   def __init__(self, environment: EnvironmentWithRelations, stop_event: EventClass, daemon: bool = False):
        super().__init__(stop_event=stop_event, daemon=daemon)
        self.environment_config = environment
      
   async def lifecycle(self):
        # one robot for now until refactor of environment
        try:
            robot = self.environment_config.robots[0]

            #logic to get robot from config:
            follower = self._robot_worker(robot)
            leader = None
            if robot.tele_operator:
                leader = await self._robot_worker(robot.tele_operator.robot)

            cameras = { camera.name, await self._camera_worker(camera):  for camera in self.environment_config.cameras}

            yield EnvironmentContext(
                follower=follower,
                leader=leader,
            )
        except Exception as e:
            logger.exception(f"Unhandled exception in {self.name}")
            raise e
        finally:
            follower.disconnect()
            if leader:
                leader.disconnect()
            
    async def run_loop(self, context: EnvironmentContext):
        # do things with context.
      
```

The BaseProcessWorker or BaseThreadWorker will inherit from thread or mp.process and critically the run function will do this:
```Python

def run(self) -> None:
    async with self.lifecycle() as context:
        self.run_loop(context)
```



## Communication and Data Store

Some of these components live in different processes for stability. This means that cross process communication needs to be established. Two will be added:

1. A Store

A store where the robot state, actions and observations, task (current joint state, next joint state, camera views respectively and task) are kept. 
These are kept in SharedMemory so that all components can easily read and write to it.
The shape of the store depends on what was loaded first.
A dataset for example has a specific feature set (robot state and cameras), the actions are simply a mirror of the robot state.
A model has specific model inputs which decide the shape and an Environment has robots and cameras which decide the robot state, actions and observations.
The store initially only has state (array), action (array)  and observations (dict of arrays).
This distills down to state being a Sample.
The Sample contains the shared memory, the feature names (list of joints or just a name in case of observation (e.g. overhead, wrist)) and a timestamp of when it was set. 

2. Events over Zenoh. 

To be able to easily handle events components need to be able to listen for specific events over zenoh. The consumer of the session (e.g. a websocket connection from fastapi) needs to be able to adjust things in the subcomponents. 

## Workers

Most components run on their own frequency. A dataset can be recorded at 50hz, robot at 100hz and model at 30hz for example. It is then also possible to run inference using a model of 30hz but record at 50hz. Yes, there will be some stepping, but that is fine. 

### Robot

Every robot needs their own worker in a process. It connects to the robot and disconnects it after done. It runs on a specific hz and takes writes their features to Store state and reads the action from the Store and writes it to the robot using the timestamp to get the goal_time. Not every robot uses goal_time and that is fine for now. No complex interpolation

### Camera

CameraWorker is a thread worker that starts a shared_camera from physicalai library. It then writes to the Store's observation shared memory of that camera at the desired hz.

### Dataset

A dataset worker runs in a separate process. It listen for start, stop, discard recording events. Once recording it writes the store's values as a frame to the dataset. This is at a desired hz. 

### Control

In order to allow for very flexible control of the entire system we need a control system. 
1. TeleoperationControl reads from leader and writes to the store.action.
2. ModelControl takes a loaded model and reads the current action from the model giving the current store frame and writes it to the store.action.
3. MoveControl will gradually move the robot from the current position to the target position over a time by writing to the store.action.

However, it needs more flexibility than just writing to the storte.action. A future human-in-the-loop Control would have to deal with the model, leader and perhaps even set the leader position. To make the Control as powerful as possible it is ran in a seperate process or thread and it can get zenoh events. Allowing for seperate hz's for whatever purpose it needs. Only one control can be activate at the same time.

## RuntimeSession

RuntimeSession all ties this together. The RuntimeSession holds the current store. On load_environment it will do so. Detailed errors will be thrown when a load_dataset (or other components) is called with a config that does not match current feature spec of the store. All subworkers are added to the Context of the RuntimeSession so that it cleans up after itself.

---

# Implementation Plan

## Starting point

What we have today and what we keep from it:

- `src/runtime/` drives everything through `physicalai.runtime.RobotRuntime`: a single loop at one fps, with Studio plugging in an `ActionSource` and some callbacks. It runs in a detached subprocess found over zenoh (`transport/`, `owner.py`, `hosts/`). This is the code we replace.
- `physicalai.runtime` also gives us the policy execution (`PolicySource`, `ChunkedActionQueue`/`ActionChunk(t0, dt)`, `AsyncExecution`, `LerpSmoother`, RTC). Bringing this in-house is the real work behind `ModelControl`.
- We keep depending on `physicalai.robot`, `physicalai.capture` (`SharedCamera`), `physicalai.inference` (`InferenceModel`) and `physicalai.config`. Studio stops *running* on `physicalai.runtime`.
- **Runtime config export stays.** The environment and model "export runtime config" endpoints (`api/environments.py`, `api/models.py`) write a YAML that runs on `physicalai.runtime.RobotRuntime` *outside* Studio, which is the deployment path. They only build a Config document and never run it, so they move to `services/runtime_export.py` and keep their `physicalai.runtime` class paths.
  - **Why both exist:** `physicalai.runtime` stays the lean way to run a policy on a robot without Studio. Studio's own runtime has different needs (concurrent rates, swappable controls, recording, HIL), so the two are allowed to diverge internally.
  - **Export parity is a hard requirement:** whatever a user can run in Studio with `ModelControl` must stay exportable as a runnable `physicalai.runtime` config, as it is today.
    - `ModelControl`'s settings (fps, sync or async execution, chunk threshold, smoothing) are chosen so each one maps to a `physicalai.runtime` equivalent: `SyncExecution`/`AsyncExecution`, `ChunkedActionQueue`, `LerpSmoother`.
    - `services/runtime_export.py` builds the export from the same `ModelControlConfig` the session uses, not from a separate set of defaults.
    - A Studio-only setting with no runtime equivalent (e.g. HIL) is left out of the export with a `# NOT EXPORTED:` comment, the same way `CHANGE_ME` paths are flagged today.
  - **Test:** a round-trip test exports an environment + model and runs `Config.from_dict(...).instantiate()` against `physicalai.runtime` with fake devices. This catches drift when either side changes.
- The `rhecker/flexible-runtime` prototype (`runtimev2/`) already works through most of this design. We port from it instead of rewriting:
  - `shared_store.py` (shm layout, create and attach by name) becomes the base of the Store, minus its seqlock.
  - `workers/process.py` and `workers/base.py` (ready/failure propagation, kill a wedged child) become the base of the workers.
  - `environment.py`, `features.py` and `dataset_layout.py` become the feature spec, derived from an environment or a dataset.
  - `control/*` gives us the Control shape.
  - `workers/robot.py` gives us the robot loop. We drop its interpolation, since the plan says goal_time only.

## Package and branch

- Feature branch `feature/runtime-inhousing`. Each phase below is its own PR into that branch, and `main` gets one merge at the end.
- New code goes in `src/runtime_session/`.
- **The old runtime is removed first (Phase 0), not at the end.** Running the old and new runtimes side by side would mean keeping both working on one branch. Instead, features that depend on the runtime are disabled in the UI with a "work in progress" state and come back phase by phase. The feature branch may have fewer features than `main` while it is in progress; `main` only gets the result once everything is back.

## Proposed changes to the sketch

1. **The worker base is generic over its context.** `lifecycle()` is an `@asynccontextmanager` that runs inside the thread or child process. `run()` becomes:
   ```python
   def run(self) -> None:
       asyncio.run(self._main())

   async def _main(self) -> None:
       try:
           async with self.lifecycle() as context:
               self._ready.set()
               await self.run_loop(context)
       except BaseException as exc:
           self._report_failure(exc)  # text over an mp.Queue for processes
       finally:
           self._ready.set()
   ```
   The parent side uses the worker itself as an async context manager. `async with worker:` starts it, waits for `ready`, and re-raises a startup failure. Leaving the block sets `stop`, joins with a timeout, and kills the worker if it is still alive. `RuntimeSession` then holds one `AsyncExitStack` and pushes every worker onto it, which is what "added to the Context of the RuntimeSession" means.
2. **Processes build their hardware in the child.** The worker object gets pickled, so it carries config only (DB rows or Config dicts), never drivers. `lifecycle()` does the instantiation.
3. **Environment is a composite context manager, not a worker.** In the sketch, `EnvironmentWorker` has no loop of its own: the robot runs in its own process and the cameras in their own threads. `load_environment` is just an `AsyncExitStack` that enters the follower `RobotWorker`, the optional leader `RobotWorker` and the `CameraWorker`s **in parallel**. That keeps today's parallel connect, and if anything fails to connect, everything already connected is released.
4. **The Store gets a leader slot.** It stays single-robot: one environment, one robot (see Decisions). Teleop still has to read the leader from somewhere, and the leader runs in its own process, so the store holds an optional `leader_state` next to `state`. HIL will later need a `leader_action` slot as well; we add it when that control arrives.

## Store design

```
FeatureSpec
  joint_names:  tuple[str, ...]          (state and action share them)
  leader:       bool                     (whether leader_state exists)
  observations: {camera_key: ImageSpec(h, w, c, dtype=uint8)}
  task:         str  (not in shm, see below)

Store (shm, one block per session, name = f"pai-{session_id}")
  state            -> Sample(values, names, timestamp)
  action           -> Sample(values, names, timestamp)
  leader_state     -> Sample(values, names, timestamp)   (only with a leader)
  observation[cam] -> Sample(frame,  name,  timestamp)
```

The joint names come namespaced the way the dataset stores them (e.g. `shoulder_pan.pos`). Then a dataset's `observation.state` and `action` names compare directly against the spec without a mapping step.

- **No locking.** Every feature (joint vector or camera frame) is a single buffer plus a timestamp, and readers copy it out without coordinating with the writer. A read that overlaps a write can be torn: some joints or rows from the previous sample, the rest from the next. We accept that because consecutive samples are only one tick apart and the world moves continuously, so a torn sample is still physically plausible.
  - **No garbage values:** aligned 8-byte stores are atomic on x86-64 and arm64. A torn joint vector mixes two real readings and never contains half-written floats.
  - **Write order:** the writer stores the values first and the timestamp last. A reader that sees a timestamp therefore never gets the zeros of a feature that has not been written yet; NaN still means "not written yet".
  - **Fallback:** if tearing turns out to matter somewhere (for example in recorded frames), add a seqlock or extra slots for that feature only. Readers don't need to change for that.
- **Timestamps:** `time.monotonic()`, which is system-wide on Linux and comparable across processes. NaN means "not written yet".
- **Task:** too awkward to keep in shm, since it is a variable-length string. Put it in session state and broadcast changes over zenoh; the dataset and model workers cache the latest value.
- **Lifetime:** `RuntimeSession` creates and unlinks the block, and children attach by name plus spec. At backend startup, sweep `/dev/shm/pai-*` blocks whose session no longer exists, because a SIGKILLed session leaves its block behind.
- **First loaded defines the shape:**
  - an environment supplies the robot and cameras;
  - a dataset supplies the state/action names and camera shapes;
  - a model supplies its input features.

  Every later load is checked against the spec with `spec.check_compatible(other) -> list[Mismatch]`. A mismatch raises `FeatureSpecMismatchError` listing each difference, for example "camera `wrist`: dataset 480x640x3, environment 720x1280x3" or "joint names differ at index 3". Once every shape-defining component is unloaded, the store is dropped and the next load creates a fresh one.

## Event bus (zenoh)

- One zenoh session per process, peer mode, localhost only, scouting off. `eclipse-zenoh` is already a direct dependency.
- Key layout is `studio/session/{session_id}/…`:
  - `cmd/{component}`: commands into a component (pub/sub, pydantic payloads, the same discriminated-union style as `contract.py`).
  - `evt/{component}`: state and errors out of a component, which the session aggregates for the websocket.
  - `rpc/{component}/{op}`: a zenoh queryable for commands that need an ack or a result (save/discard episode, load model). It always has a timeout.
- Telemetry (joints and frames) for the UI does **not** go over zenoh. The FastAPI process attaches to the Store and samples it at UI rate. That is cheaper and removes today's bounded event queue.

## Workers

| Worker | Kind | Rate | Reads | Writes |
|---|---|---|---|---|
| `RobotWorker` (follower) | process | robot hz (e.g. 100) | `action` | `state` |
| `RobotWorker` (leader, optional) | process | robot hz | – (later: `leader_action` for HIL) | `leader_state` |
| `CameraWorker` | thread (`SharedCamera` already runs its own publisher process) | camera fps | – | `observation[cam]` |
| `DatasetWorker` | process | dataset fps | full snapshot | LeRobot frames |
| `Control` (exactly one) | thread or process, by type | its own hz | whatever it needs | `action` |

- **RobotWorker:**
  - On connect it seeds `action` with the measured position, so with no control running the arm holds still.
  - Before driving, it checks the robot's joint order against the spec; the prototype already does this.
  - **Action timestamp:** the time the action *should have happened*, which is usually now. A teleop or move control writes `now`. A model writes `t0 + i*dt` for the step it samples.
  - **goal_time:** each tick, `goal_time = max(action.timestamp - previous_tick_timestamp, 1/robot_hz)`. The `1/hz` floor stops the robot from being asked to jump instantly, so in practice goal_time will mostly be `1/hz`. There is no interpolation, and robots without goal_time ignore it.
  - Any timeout or exception ends the worker. The session sees the child die and emits an error. There are no silent retries.
- **DatasetWorker:**
  - Commands are start, stop, save and discard; save and discard are RPCs and acked.
  - **Join timeout ≥ 60 s.** Finalizing copies the cache back to the dataset, and killing the process mid-copy loses the dataset. This is the main lesson to carry over from `session.py` today, along with "discard the open episode on teardown".
- **Controls:**
  - `MoveControl` interpolates from the current state to a target over T and then holds. It is also the right thing to run when switching between controls.
  - `TeleopControl` copies `leader_state` to `action`.
  - `ReplayControl` plays a loaded dataset's episode into `action` at the dataset fps, writing each frame's timestamp as `start + frame_index / fps`. It needs a loaded dataset and an environment whose spec matches it.
  - `ModelControl` (process) snapshots the store into an observation, runs inference with `t0 = observation timestamp`, and samples a clock-indexed chunk queue at control hz. This is where the time-based `ActionChunk` work is ported over. Order: sync first, then async with a separate inference thread, then smoothing, and RTC last.
- **Switching control:** stop the old control (join), re-seed `action` from the current state, then start the new one. That keeps the one-active-control rule structural instead of arbitrated.

## RuntimeSession API (asyncio, in the FastAPI process)

```python
class RuntimeSession:
    async def load_environment(self, env: EnvironmentWithRelations) -> None
    async def unload_environment(self) -> None
    async def load_dataset(self, dataset: Dataset) -> None
    async def unload_dataset(self) -> None
    async def load_model(self, model: Model) -> None
    async def set_control(self, control: ControlConfig | None) -> None  # discriminated: hold/teleop/model/move/replay
    async def set_task(self, task: str) -> None
    def store(self) -> StoreReader
    async def aclose(self) -> None  # unwinds the AsyncExitStack in reverse
```

- **At most one environment at a time.** Calling `load_environment` while another is loaded raises an error; the caller unloads first. An environment has exactly one robot for now (`robots[0]`), and loading one with more robots raises an error that says so.
- **Lifetime:** the session lives in the FastAPI process and **belongs to the websocket that opened it**. Closing the socket calls `aclose()`, and backend shutdown (`core/lifecycle.py`) closes whatever is left. Nothing is reattached: a page reload opens a new session. The `RuntimeSessionRegistry` only lists live sessions and enforces claims.
- **Claims:** the registry enforces exclusive claims on robots and cameras across sessions. It reuses `camera_claims` and reports `RuntimeSessionBusyError` to the setup and calibration workers.

## UI

### Today's edges

| Surface | Backend today | Problem |
|---|---|---|
| Camera preview (camera form, environment form, record and inference views) | `/api/cameras/ws`, one `CameraWorker` and one socket **per camera** | Record and inference pages open N camera sockets next to the session socket, so frames and joints come from different places at different rates. |
| Robot detail and robot form | `/robots/{id}/observations/ws`, a `SharedRobot` attach | Separate code path, and its own copy of the error mapping. |
| Environment preview (`robot-cell.tsx`) | `useJointState` → `/runtime/ws?follower_id=` (detached session keyed by follower) | A preview starts a full control session and relies on reattach and idle timeouts. |
| Record and inference | `RuntimeSessionProvider` → same `/runtime/ws`, plus camera sockets | Reattach and handshake logic in `runtime-session-provider.tsx` and `use-joint-state.ts`. |

### Target

Three levels, each a strict superset of the one below it. Every one uses the same workers and the same Store, so preview, recording and inference all run the same code.

1. **Single camera.** `/api/cameras/ws` opens a `CameraWorker` (and so a `SharedCamera`) writing into a one-camera store, and streams JPEG at the camera fps. It's used by the camera form.
2. **Single robot.** `/robots/{id}/observations/ws` opens a `RobotWorker` in **observe-only mode** (it never sends an action, never changes torque) writing into a one-robot store, and streams its joints. It's used by the robot detail page and the robot form.
3. **Runtime session.** `/api/projects/{project_id}/runtime/ws` creates a `RuntimeSession` for the lifetime of the socket. The UI sends commands to load things into it:
   - the environment form preview loads **only the environment**;
   - the record page loads environment + dataset;
   - the inference page loads environment + model.

**Levels 1 and 2 are for the forms only.** Pages that have a session (environment preview, record, inference) **never** open a camera or robot socket. Their camera and joint views read from the `RuntimeSessionProvider`'s refs, so a robot is connected exactly once, by the session's `RobotWorker`.

**Device already held by a session:** a level 1 or 2 socket asking for it gets a clear `device_in_use` error naming the session, instead of a second connection. Reading from the session's store would let a form preview a held device, but it is not needed for the forms and we can add it later.

### Session websocket protocol

Client → server: JSON commands, each with a `request_id`. **Every** command gets an `ack` with `ok` and either an error or a result, which removes today's split between fire-and-forget and acked commands.

```
load_environment {environment_id}   unload_environment
load_dataset {dataset_id}           unload_dataset
load_model {model_id, device}       unload_model
set_control {type: hold|teleop|model|move|replay, ...}
set_task {task}
start_episode  stop_episode  save_episode  discard_episode
```

Server → client:

- `state` (JSON, sent on change): what is loaded, which control is active, recording status, episodes recorded, and the store spec, so the UI knows the joint and camera names.
- `error` (JSON): component, message, error_code, and whether the session survived it. A robot worker dying is fatal to the environment, not to the socket.
- `observation` (JSON, **~20 Hz**, configurable via `?hz=`): `{timestamp, state: {joint: value}, action: {...}, leader_state: {...}}`, read from the store.
- **Camera frames (binary, same 20 Hz tick):** `[u8 key_len][key utf-8][jpeg bytes]`, one message per camera per tick. Only frames newer than the last one sent get re-encoded, so a 15 fps camera isn't encoded 20 times a second. Binary avoids the ~33% base64 overhead.

JPEG encoding runs off the event loop (`asyncio.to_thread`). One sampler task per session reads the store and fans out to the socket.

### Frontend

- **One `RuntimeSessionProvider`, rewritten.**
  - It opens the socket and sends the loads its props imply (environment, then optional dataset and model).
  - It exposes `state`, typed command mutations (one promise per ack), and refs for observation and frames.
  - It drops the handshake, reattach and idle logic in today's `runtime-session-provider.tsx` and `use-joint-state.ts`.
- **20 Hz data stays out of React state.** Observations and frames go into refs or a small external store (`useSyncExternalStore`). Each camera renders by drawing an `ImageBitmap` onto a canvas, so a tick does not re-render the page.
- **`CameraView` and `RobotJointsView` are dumb components** that take their data from a source:
  - inside a provider, `useSessionCamera(key)` and `useSessionJoints()` read the provider's observation and frame refs, and no extra socket is opened;
  - on the forms, `useCameraSocket(camera)` and `useRobotSocket(robot)` use the standalone sockets.

  That removes the duplicated `camera-cell.tsx` and `camera-cell.component.tsx`. The record and inference viewers drop their per-camera `WebsocketCamera` sockets.
- The environment preview becomes `<RuntimeSessionProvider environment={env}>` with the existing cells. The record page adds `dataset`, and the inference page adds `model`.
- The runtime-sessions footer and list come back on the registry in Phase 5. "Stop" closes the session, and the owning socket receives a final `error` with `session_closed`.
- Regenerate the OpenAPI types at each phase that changes a schema. The ws message types become pydantic models included in the OpenAPI schema, so the UI gets them typed.

## Phases (each one a PR into the feature branch)

0. **Remove the current runtime.**
   - Backend:
     - Delete `src/runtime/` (session, owner, hosts, transport, action source, callbacks, policy loader), `api/runtime_ws.py`, `api/runtime_sessions.py`, `services/runtime_session_service.py`, `schemas/runtime_session.py` and their routers in `main.py`.
     - Delete `tests/runtime/` and the matching `tests/api/` files: runtime ws, runtime sessions, runtime owner, robot control errors and request ack.
     - Delete the stale `tests/runtimev2/` and the leftover `src/runtimev2/__pycache__`.
   - **Moved, not deleted:**
     - `runtime/features.py` (`observation_to_dict`, `sanitize_camera_name`, …) moves to `robots/features.py`. It is used by `robots/physicalai_adapter.py`, `api/robot_observations.py` and the zero-calibration worker.
     - `ErrorEvent` and `ObservationEvent` move into a small `schemas/robot_stream.py` for `robot_observations`.
     - The runtime config export (`build_runtime_config`, `policy_source_fragment`, `runtime_export_readme`, `runtime_config_change_me`) moves to `services/runtime_export.py`. See the note below.
   - `api/robots.py`: drop the `runtime_session_holder` busy check for now. Registry claims replace it in Phase 5.
   - Untouched: `/api/cameras/ws` and `/robots/{id}/observations/ws` do not depend on the runtime and keep working until Phase 4 replaces them.
   - UI:
     - Delete `runtime-session-provider.tsx`, `use-joint-state.ts` and the runtime-sessions feature, and remove the footer badge.
     - Replace the environment form preview's robot cell with a "Live preview is being reworked" placeholder. Camera cells keep their standalone socket for now.
     - Record and inference pages: put a work-in-progress empty state behind a single `RUNTIME_WIP` flag; their buttons (record, run model, teleop) are disabled.
     - Regenerate the OpenAPI types.
   - The rest of the app has to keep working: robots, cameras, environments CRUD, datasets, training and export. `prek` and the backend and UI tests must pass.
   - **Done.** Deviations from the steps above:
     - `RuntimeSessionBusyError` (backend) and `isRuntimeSessionBusyError` (UI) were removed too; Phase 5 adds the claim errors back.
     - `RUNTIME_IDLE_TIMEOUT_S` was removed from settings.
     - `useSynchronizeModelJoints` moved out of `use-joint-state.ts` into `use-synchronize-model-joints.ts`.
     - The record and inference viewers (`recording-viewer.tsx`, `inference-viewer.tsx`) and `features/robots/robot-control/` were deleted rather than kept, because they only worked through the old provider. Phases 7 and 8 rebuild them on the new provider; the old layout is in git history.
     - `RUNTIME_WIP` lives in `components/runtime-wip/runtime-wip.tsx`, together with `RuntimeWipNotice`. It disables "Start recording", "Add Episode" and "Run model". The environment preview shows the robot model at rest with an info note.
1. **Worker foundations:** generic `ThreadWorker[C]` and `ProcessWorker[C]` in `src/workers/base.py`, replacing the old `BaseThreadWorker`/`BaseProcessWorker`, plus `Rate` in `src/workers/rate.py` replacing `run_at_frequency`. Covers readiness and failure propagation, stop, join, then kill. The existing workers (training, dataset import, training dispatcher, camera) and the `Scheduler` moved onto it. Tests use fake workers that crash during setup, crash mid-run, and hang.
2. **Store:** `FeatureSpec`, `Sample`, lock-free scalar and image buffers (values first, timestamp last), create and attach, compatibility checks with detailed mismatches, and the stale-block sweep. Includes a cross-process read/write test.
3. **Event bus:** a typed zenoh wrapper with pub/sub and an RPC with ack and timeout. Tests run against a local-only zenoh config.
4. **Robot and camera workers, plus the environment:** spec from the environment, ported from `runtimev2/environment.py`, and observe-only mode for `RobotWorker`. Move `/api/cameras/ws` and `/robots/{id}/observations/ws` onto the new workers (UI levels 1 and 2). The frontend gets the dumb `CameraView` and `RobotJointsView` plus the standalone hooks. Tested with a fake robot driver, then once on a real SO-101 through `scratch.py`.
5. **RuntimeSession, registry and session websocket:** load and unload the environment, store lifetime, claims (`device_in_use` for levels 1 and 2, the busy check in `api/robots.py`). Includes the new ws protocol at `/runtime/ws` (commands with acks, `state`, 20 Hz `observation`, binary frames), the new `RuntimeSessionProvider`, and the runtime-sessions footer and list on the registry. The **environment form preview** comes back: its placeholder is removed and its cells read from the provider.
6. **Controls:** base class, `MoveControl`, `TeleopControl`, and the switching protocol. Teleop buttons are re-enabled.
7. **Dataset worker:** spec from the dataset (it can be the first load), recording commands, acked save and discard, the finalize path. The **record page** comes back on the new provider (lifting `RUNTIME_WIP` for it). `ReplayControl` follows in the same phase.
8. **ModelControl:** port the policy execution off `physicalai.runtime` (sync, async with the time-based chunk queue, smoothing). Introduce `ModelControlConfig` and switch `services/runtime_export.py` over to it, along with the export round-trip test. The **inference page** comes back on the new provider, and the `RUNTIME_WIP` flag is deleted. Measure with the oracle model: `actions[i] = f(t0 + i)`.
9. **Finish:**
   - Check that no `physicalai.runtime` import is left outside `services/runtime_export.py`.
   - Rewrite `docs/explanation/runtime-session-architecture.md`.
   - Do a full manual pass on real hardware: preview, teleop, record + save/discard, inference, replay.
10. **Merge** the feature branch into `main`.

## Decisions

1. **RuntimeSession lives in the FastAPI process** for now and belongs to its websocket. Closing the socket or restarting the backend ends the session, and there is no reattaching.
2. **One active environment per session, with one robot per environment** until the environment refactor. The robot may have one leader.
3. **The action timestamp is the time the action should have happened** (usually now). The robot worker sets `goal_time = max(action.timestamp - previous_tick_timestamp, 1/hz)`.
4. **A loaded dataset decides the store shape**, just as an environment or model would. Replaying episodes is a Control (`ReplayControl`).
5. **Single robot until the environment refactor.** The store holds one `state` and one `action`, plus `leader_state`.
6. **UI levels:** a camera form opens a `CameraWorker`, a robot form opens an observe-only `RobotWorker`, and the environment, record and inference pages open a `RuntimeSession`, loading environment only, + dataset, or + model respectively. All live updates come from the session store at ~20 Hz. Pages with a session read joints and frames only from the provider, never from per-device sockets.
7. **Keep the runtime export.** Studio exports a config that runs on `physicalai.runtime` outside Studio, built from the same `ModelControlConfig` the session runs. Settings without a runtime equivalent are flagged as not exported.
8. **Remove first, rebuild after.** The old runtime is deleted in Phase 0; unfinished features show as work in progress on the feature branch until their phase lands.
